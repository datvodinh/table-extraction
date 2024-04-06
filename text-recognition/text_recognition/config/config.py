from argparse import Namespace
from dataclasses import dataclass


@dataclass
class SwinTransformerOCRConfig:
    # MODEL
    in_channels: int = 3
    num_workers: int = 2
    lr: float = 0.00001
    betas: tuple[float] = (0.9, 0.98)
    eps: float = 1e-09
    # ENCODER
    patch_size: tuple[int] = (4, 4)
    embed_dim: int = 96
    window_size: tuple[int] = (8, 8)
    depths: tuple[int] = (2, 6, 2)
    num_heads: tuple[int] = (6, 12, 24)
    # DECODER
    d_model: int = 384
    decoder_layers: int = 6
    decoder_attention_heads: int = 6
    decoder_ffn_dim: int = 2048
    scale_embedding: bool = True
    dropout: float = 0.1
    # INFERENCE
    max_tokens: int = 128
    # DATAMODULE
    train_ratio: float = 0.99
    img_size: tuple[int] = (32, 512)
    pct_start = 0.3
    label_smoothing: float = 0.1
    max_grad_norm: float = 1.0
    batch_size: int = 2
    max_epochs: int = 10
    seed: int = 0

    def update(
        self,
        args: Namespace
    ):
        if isinstance(args, dict):
            args = Namespace(**args)
        for k, v in args.__dict__.items():
            if hasattr(self, k) and v is not None and k != "stage":
                setattr(self, k, v)
        print("Config Updated!")
