import torch
import cv2
import numpy as np
from PIL import Image
from transformers import TableTransformerForObjectDetection
from dataclasses import dataclass
from torchvision import transforms
from typing import Any


class MaxResize:
    def __init__(self, max_size=1000):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))

        return resized_image


class TableStructureRecognitionModel:
    def __init__(self, device="cpu") -> None:
        self.model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-structure-recognition-v1.1-all"
        )
        self.model.to(device)

        self.transform = transforms.Compose([
            MaxResize(800),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.device = device
        self.id2label = self.model.config.id2label
        self.id2label[len(self.model.config.id2label)] = "no object"

    @torch.no_grad()
    def predict(self, image: Image.Image):
        pixel_values = self.transform(image).unsqueeze(0)
        outputs = self.model(pixel_values)
        return self._outputs_to_objects(outputs, image.size, self.id2label)

    def _box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def _rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self._box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def _outputs_to_objects(self, outputs, img_size, id2label):
        m = outputs.logits.softmax(-1).max(-1)
        pred_labels = list(m.indices.cpu().numpy())[0]
        pred_scores = list(m.values.cpu().numpy())[0]
        pred_bboxes = outputs['pred_boxes'].cpu()[0]
        pred_bboxes = [
            elem.tolist()
            for elem in self._rescale_bboxes(pred_bboxes, img_size)
        ]

        objects = []
        for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
            class_label = id2label[int(label)]
            if not class_label == 'no object':
                objects.append(
                    {
                        'label': class_label,
                        'score': float(score),
                        'box': {
                            'xmin': bbox[0]-1,
                            'ymin': bbox[1]-1,
                            'xmax': bbox[2]+1,
                            'ymax': bbox[3]+1
                        }
                    }
                )

        return objects


@dataclass
class TableStructureRecognitionOutput:
    results: list[dict[str, Any]]
    images: np.ndarray


class TableStructureRecognitionInference:
    def __init__(self, device: str = "cpu") -> None:
        self.model = TableStructureRecognitionModel(device=device)

    def _process_image(self, data: np.ndarray | Image.Image | str) -> Image.Image:
        if isinstance(data, Image.Image):
            image = data
        elif isinstance(data, str):
            image = Image.open(data)
        elif isinstance(data, np.ndarray):
            image = Image.fromarray(data)

        return image

    def _get_prediction(self, image: np.ndarray):
        return self.model.predict(image)

    def _get_bounding_boxes(self, images: np.ndarray, results: list[str, str, dict]):
        if isinstance(images, Image.Image):
            images = np.array(images)
        for res in results:
            box = res['box']
            left, top = int(box['xmin']), int(box['ymin'])
            right, bottom = int(box['xmax']), int(box['ymax'])
            crop_image = images[top:bottom, left:right]
            res["image"] = crop_image
        return results

    def _draw_bounding_boxes(self, images: np.ndarray, results: list[str, str, dict]):
        if isinstance(images, Image.Image):
            images = np.array(images)
        for res in results:
            score, label, box = res['score'], res['label'], res['box']
            left, top = int(box['xmin']), int(box['ymin'])
            right, bottom = int(box['xmax']), int(box['ymax'])
            label = f"{label}: {round(score, 3)}"
            imgHeight, imgWidth, _ = images.shape
            thick = int((imgHeight + imgWidth) // 900)
            cv2.rectangle(images, (left, top), (right, bottom), (0, 255, 0), thick)
        return images

    def predict(self, data: np.ndarray | Image.Image | str) -> TableStructureRecognitionOutput:
        image = self._process_image(data)
        predict = self._get_prediction(image)
        results = self._get_bounding_boxes(image, predict)
        images = self._draw_bounding_boxes(image, predict)
        return TableStructureRecognitionOutput(
            results=results,
            images=images
        )
