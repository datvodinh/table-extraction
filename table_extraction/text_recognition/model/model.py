import torch
from .swin import SwinTransformerOCR
from .vit import VisionTransformerOCR
from text_recognition.config import SwinTransformerOCRConfig, VisionTransformerOCRConfig


class TextRecognitionModel:

    @staticmethod
    def from_config(config: SwinTransformerOCRConfig | VisionTransformerOCRConfig):
        if isinstance(config, SwinTransformerOCRConfig):
            return SwinTransformerOCR(config)
        elif isinstance(config, VisionTransformerOCRConfig):
            return VisionTransformerOCR(config)
        else:
            raise ValueError("Invalid config!")

    @staticmethod
    def from_path(ckpt_path: str):
        try:
            config = torch.load(
                ckpt_path, map_location=torch.device('cpu')
            )['hyper_parameters']['config']
            if isinstance(config, SwinTransformerOCRConfig):
                return SwinTransformerOCR.load_from_checkpoint(ckpt_path)
            elif isinstance(config, VisionTransformerOCRConfig):
                return VisionTransformerOCR.load_from_checkpoint(ckpt_path)
            else:
                raise ValueError("Invalid config!")
        except ValueError:
            raise ValueError("Wrong Checkpoint Path!")
