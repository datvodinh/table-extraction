import pytesseract
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Any
from PIL import Image


@dataclass
class TextDetectionOutput:
    results: list[dict[str, Any]]
    images: np.ndarray


class TextDetectionInference:
    def __init__(self) -> None:
        pass

    def _process_image(self, data: np.ndarray | Image.Image | str):
        if isinstance(data, Image.Image):
            image = np.asarray(data)
        elif isinstance(data, str):
            image = cv2.imread(data)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(data, np.ndarray):
            image = data

        return image

    def _get_prediction(self, image: np.ndarray):
        return pytesseract.image_to_data(
            image=image,
            output_type=pytesseract.Output.DICT
        )

    def _get_bounding_boxes(self, image: np.ndarray, predict: dict):
        results = []
        for i in range(len(predict["text"])):
            if int(predict["conf"][i]) > 0:
                res = {}
                res['xmin'] = predict["left"][i]
                res['ymin'] = predict["top"][i]
                res['xmax'] = predict["width"][i] + res['xmin']
                res['ymax'] = predict["height"][i] + res['ymin']
                left, top = int(res['xmin']), int(res['ymin'])
                right, bottom = int(res['xmax']), int(res['ymax'])
                crop_image = image[top:bottom, left:right]
                res["image"] = crop_image
                results.append(res)
        return results

    def _draw_bounding_boxes(self, image: np.ndarray, predict: dict):
        for i in range(len(predict["text"])):
            x = predict["left"][i]
            y = predict["top"][i]
            w = predict["width"][i]
            h = predict["height"][i]
            conf = int(predict["conf"][i])
            if conf > 0:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return image

    def predict(self, data: np.ndarray | Image.Image | str) -> TextDetectionOutput:
        image = self._process_image(data)
        predict = self._get_prediction(image)
        results = self._get_bounding_boxes(image, predict)
        images = self._draw_bounding_boxes(image, predict)
        return TextDetectionOutput(
            results=results,
            images=images
        )
