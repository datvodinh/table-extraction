from dataclasses import dataclass
from argparse import Namespace


class TransformerOCRConfig:
    # MODEL
    in_channels: int = 3
    embed_dim: int = 768
    num_workers: int = 4
    num_patches: tuple[int, int] = (4, 8)
    lr: float = 0.00001
    # DATAMODULE
    train_ratio: float = 0.95
    img_size: tuple[int, int] = (32, 128)
    pct_start = 0.3
    label_smoothing: float = 0.1
    max_grad_norm: float = 0.5
    batch_size: int = 2
    max_epochs: int = 10
    seed: int = 0
    special_tokens: dict[str, str] = {
        "pad_token": "<|PAD|>",
        "sep_token": "<|SEP|>"
    }
    tokenizer_len: int = 50257 + len(list(special_tokens.keys()))

    def update(
        self,
        args: Namespace
    ):
        if isinstance(args, dict):
            args = Namespace(**args)
        for k, v in args.__dict__.items():
            if hasattr(self, k) and v is not None and k != "stage":
                setattr(self, k, v)
        print("LDM Config Updated!")