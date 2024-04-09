from argparse import Namespace
from dataclasses import dataclass


@dataclass
class SwinTransformerOCRConfig:
    # MODEL
    in_channels: int = 3
    num_workers: int = 2
    lr: float = 0.0003
    betas: tuple[float] = (0.9, 0.98)
    eps: float = 1e-09
    div_factor: int = 20
    final_div_factor: int = 100
    # ENCODER
    patch_size: tuple[int] = (4, 4)
    embed_dim: int = 96
    window_size: tuple[int] = (8, 8)
    depths: tuple[int] = (2, 12, 2)
    num_heads: tuple[int] = (6, 12, 24)
    # DECODER
    d_model: int = 384
    decoder_layers: int = 6
    decoder_attention_heads: int = 6
    decoder_ffn_dim: int = 1536
    scale_embedding: bool = False
    dropout: float = 0.1
    use_learned_position_embeddings: bool = True
    activation_function: str = "silu"
    # INFERENCE
    max_tokens: int = 128
    # DATAMODULE
    train_ratio: float = 0.99
    img_size: tuple[int] = (64, 256)
    pct_start = 0.1
    label_smoothing: float = 0.1
    max_grad_norm: float = 0.5
    batch_size: int = 32
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
