from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import torch
import cv2


class TableDectectionModel:
    def __init__(self, device="cpu") -> None:
        self.processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
        self.model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
        self.device = device
        self.model.to(device)

    def _preprocess_image(self, images: Image):
        """
        Return:
        images = {
            'pixel_values': torch.Tensor,
            'pixel_mask': torch.Tensor
        }
        """
        return self.processor(
            images=images, return_tensors="pt"
        ).to(self.device)

    def _postprocess_output(self, outputs: torch.Tensor, img_size: tuple):
        """
        Return:
        results = {
            'scores': torch.tensor([0.9,0.99,...])
            'labels': torch.tensor([0,1,...])
            'boxes': torch.tensor([[ 19.1826, 134.6679, 668.1139, 398.2249], ...]
        }

        'boxes' is coordinates [(left, top), (right, bottom)]
        """
        img_size = torch.tensor([img_size])
        return self.processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=img_size)[0]

    def predict(self, image: Image):
        images = self._preprocess_image(image)
        outputs = self.model(**images)
        results = self._postprocess_output(outputs, image.size[::-1])
        return results


class TableDetectionInference:
    def __init__(self, device: str = "cpu") -> None:
        self.model = TableDectectionModel(device)
        self.model_config = self.model.model.config
        self.box_color = (0, 255, 0)

    def _draw_bounding_boxes(self, image, results):
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            left, top = int(box[0]), int(box[1])
            right, bottom = int(box[2]), int(box[3])
            label = f"{self.model_config.id2label[label.item()]}: {round(score.item(), 3)}"
            imgHeight, imgWidth, _ = image.shape
            thick = int((imgHeight + imgWidth) // 900)
            cv2.rectangle(image, (left, top), (right, bottom), self.box_color, thick)
            cv2.putText(image, label, (left, top - 12), 0, 1e-3 * imgHeight, self.box_color, thick//3)
        return image

    def __call__(self, img_path: str):
        image = Image.open(img_path).convert("RGB")
        image_numpy = cv2.imread(img_path)
        image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)
        results = self.model.predict(image)
        image_numpy = self._draw_bounding_boxes(image_numpy, results)
        return image_numpy
