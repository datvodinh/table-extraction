from .swin_config import SwinTransformerOCRConfig
from .vit_config import VisionTransformerOCRConfig


class TextRecognitionConfig:
    def from_args(model_name: str):
        if model_name == "swin":
            return SwinTransformerOCRConfig()
        elif model_name == "vit":
            return VisionTransformerOCRConfig()
        else:
            raise ValueError(f"model_name must be in ['swin, 'vit'], have {model_name}")
