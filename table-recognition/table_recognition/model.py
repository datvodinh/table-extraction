import cv2
import numpy as np
from PIL import Image
from transformers import pipeline


class TableStructureRecognitionModel:
    def __init__(self, device="cpu") -> None:
        self.pipeline = pipeline(
            task="object-detection",
            model="microsoft/table-transformer-structure-recognition",
            device=device
        )

    def predict(self, image: Image.Image):
        """
        Returns:
        results = [
            {
                'score': 0.9984303116798401,
                'label': 'table column',
                'box': {'xmin': 405, 'ymin': 49, 'xmax': 448, 'ymax': 550}
            },
            ...
            ]
        """
        return self.pipeline.predict(image)


class TableStructureRecognitionInference:
    def __init__(self, device: str = "cpu") -> None:
        self.model = TableStructureRecognitionModel(device=device)

    def _pil_to_numpy(self, image: Image.Image):
        return np.asarray(image)

    def _numpy_to_pil(self, image: np.ndarray):
        return Image.fromarray(image)

    def _get_bounding_boxes(self, images: np.ndarray, results: list[str, str, dict]):
        for res in results:
            box = res['box']
            left, top = int(box['xmin']), int(box['ymin'])
            right, bottom = int(box['xmax']), int(box['ymax'])
            crop_image = images[top:bottom, left:right]
            res["image"] = crop_image
        return results

    def _draw_bounding_boxes(self, images: np.ndarray, results: list[str, str, dict]):
        for res in results:
            score, label, box = res['score'], res['label'], res['box']
            left, top = int(box['xmin']), int(box['ymin'])
            right, bottom = int(box['xmax']), int(box['ymax'])
            label = f"{label}: {round(score, 3)}"
            imgHeight, imgWidth, _ = images.shape
            thick = int((imgHeight + imgWidth) // 900)
            cv2.rectangle(images, (left, top), (right, bottom), (0, 255, 0), thick)
        return images

    def predict(self, image: Image.Image | np.ndarray):
        """
        Returns:
        results = [
            {
                'score': 0.9984303116798401,
                'label': 'table column',
                'box': {'xmin': 405, 'ymin': 49, 'xmax': 448, 'ymax': 550},
                'image': np.array([...])
            },
            ...
            ]
        """
        if isinstance(image, np.ndarray):
            image = self._numpy_to_pil(image)
        results = self.model.predict(image)
        return self._get_bounding_boxes(
            images=self._pil_to_numpy(image),
            results=results
        )

    def predict_visualize(self, image: Image.Image):
        """
        Returns:
        image = np.array([])
        """
        if isinstance(image, np.ndarray):
            image = self._numpy_to_pil(image)
        results = self.model.predict(image)
        return self._draw_bounding_boxes(
            images=self._pil_to_numpy(image),
            results=results
        )
