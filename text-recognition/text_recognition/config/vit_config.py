from argparse import Namespace
from dataclasses import dataclass


@dataclass
class VisionTransformerOCRConfig:
    in_channels: int = 3
    patch_size: int = 4
    max_tokens: int = 32
    num_workers: int = 2
    lr: float = 0.0003
    betas: tuple[float] = (0.9, 0.98)
    eps: float = 1e-09
    div_factor: int = 20
    final_div_factor: int = 100
    train_ratio: float = 0.99
    img_size: tuple[int] = (64, 256)
    pct_start = 0.1
    label_smoothing: float = 0
    max_grad_norm: float = 0.5
    batch_size: int = 32
    max_batch_size: int = 256
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
