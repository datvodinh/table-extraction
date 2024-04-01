import torch
import torch.nn as nn
import pytorch_lightning as pl
from einops import rearrange
from transformers import GPT2Model
from text_recognition.config import TransformerOCRConfig
from torch.optim.lr_scheduler import OneCycleLR


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_patches: tuple[int, int] = (4, 8),
        img_size: tuple[int, int] = (32, 256),
    ):
        super().__init__()
        assert (img_size[0] % num_patches[0]) == 0 and (img_size[1] % num_patches[1] == 0), \
            'Image sizes must be divisible by num patches'
        patch_size = tuple(img_size[i] // num_patches[i] for i in range(2))
        total_patches = num_patches[0] * num_patches[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, total_patches, embed_dim))
        nn.init.xavier_uniform_(self.pos_embed)
        self.linear_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )

    def forward(self, x):
        x = self.linear_embed(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return x + self.pos_embed


class GPT2Decoder(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 768,
        tokenizer_len: int = 50257,
        num_patches: tuple[int, int] = (4, 8),
        img_size: tuple[int, int] = (32, 256)
    ):
        super().__init__()
        self._patch_embed = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_patches=num_patches,
            img_size=img_size
        )

        self._model = GPT2Model.from_pretrained('gpt2')
        self.fc = nn.Linear(embed_dim, tokenizer_len)
        self._model.resize_token_embeddings(tokenizer_len)

    def _word_embed(self, x: torch.Tensor):
        pos = torch.arange(0, x.shape[-1], dtype=torch.long, device=x.device)  # shape (t)
        return self._model.wte(x) + self._model.wpe(pos)

    def _decoder_blocks(self, x: torch.Tensor):
        x = self._model.drop(x)
        for block in self._model.h:
            x = block(hidden_states=x)[0]
        x = self._model.ln_f(x)
        return self.fc(x)

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ):
        image_embed = self._patch_embed(images)  # B P E
        text_embed = self._word_embed(labels)  # B S E
        merge_inputs = torch.cat([image_embed, text_embed], dim=1)
        img_attn_mask = torch.ones(
            (text_embed.shape[0], text_embed.shape[1]), device=text_embed.device
        )
        full_attn_mask = torch.cat([img_attn_mask, attention_mask], dim=1)
        return self._decoder_blocks(merge_inputs, attention_mask=full_attn_mask)


class DecoderOnlyTransformerOCR(pl.LightningModule):
    def __init__(self, config: TransformerOCRConfig) -> None:
        super().__init__()
        self.config = config
        self.model = GPT2Decoder(
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            tokenizer_len=config.tokenizer_len,
            num_patches=config.num_patches,
            img_size=config.img_size
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

        self.save_hyperparameters()

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ):
        return self.model(images, labels, attention_mask)

    def _forward_step(self, batch: tuple, batch_idx: int, stage: str = "train"):
        images, labels, labels_shifted, attn_mask = batch
        logits = self.forward(images, labels, attn_mask)
        logits = logits[:, images.shape[1]:, :]
        loss = self.criterion(logits, labels_shifted)
        acc = torch.mean(torch.argmax(logits, dim=-1) == labels_shifted)
        self.log_dict({
            f"loss_{stage}": loss,
            f"acc_{stage}": acc
        })
        return loss

    def training_step(self, batch: tuple, batch_idx: int):
        return self._forward_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._forward_step(batch, batch_idx, stage="val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.config.lr,
        )
        scheduler = {
            'scheduler': OneCycleLR(
                optimizer=optimizer,
                max_lr=self.config.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=self.config.pct_start
            ),
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]
