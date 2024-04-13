import torch
import cv2
import numpy as np
from PIL.Image import Image
from ..datamodule import OCRTransform
from ..model import TextRecognitionModel


class TextRecognitionInference:
    def __init__(self, ckpt_path: str) -> None:
        self.model = TextRecognitionModel.from_path(ckpt_path)
        self.model.eval()
        self.transform = OCRTransform(img_size=self.model.config.img_size, stage="val")

    def _predict_from_path(self, img_path: str):
        image = cv2.imread(img_path)
        return self._from_numpy(image)

    def _predict_from_pil_image(self, img: Image):
        image = np.asarray(img)
        return self._from_numpy(image)

    def _predict_from_numpy(self, img: np.ndarray):
        image = self.transform(image=img).unsqueeze(0)
        return self.model.predict(image)

    @torch.no_grad()
    def predict(self, data: str | Image | np.ndarray):
        if isinstance(data, str):
            return self._predict_from_path(data)
        elif isinstance(data, Image):
            return self._predict_from_pil_image(data)
        elif isinstance(data, np.ndarray):
            return self._predict_from_numpy(data)
        else:
            raise ValueError("Invalid data!")
