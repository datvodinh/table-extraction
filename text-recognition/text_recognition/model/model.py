from typing import Any
from .swin import SwinTransformerOCR
from .vit import VisionTransformerOCR
from text_recognition.config import SwinTransformerOCRConfig, VisionTransformerOCRConfig


class TextRecognitionModel:

    @staticmethod
    def from_config(config: Any):
        if isinstance(config, SwinTransformerOCRConfig):
            return SwinTransformerOCR(config)
        elif isinstance(config, VisionTransformerOCRConfig):
            return VisionTransformerOCR(config)
        else:
            raise ValueError("Invalid config!")
