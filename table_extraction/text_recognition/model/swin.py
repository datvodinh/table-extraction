import torch
import torch.nn as nn
import pytorch_lightning as pl
from einops import rearrange
from torch.optim.lr_scheduler import OneCycleLR
from transformers import (
    VisionEncoderDecoderModel,
    Swinv2ForImageClassification,
    Swinv2Config
)
from text_recognition.config import SwinTransformerOCRConfig
from text_recognition.tokenizer import OCRTokenizer


class SwinTransformerEncoder(nn.Module):
    def __init__(self, config: SwinTransformerOCRConfig):
        super().__init__()
        config_pretrained = Swinv2Config.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        config_pretrained.image_size = config.img_size
        config_pretrained.num_channels = config.in_channels
        config_pretrained.num_labels = 0
        self.model = Swinv2ForImageClassification.from_pretrained(
            "microsoft/swinv2-tiny-patch4-window8-256",
            config=config_pretrained
        )
        self.model.swinv2.embeddings.patch_embeddings.projection = nn.Conv2d(3, 192, kernel_size=(4, 4), stride=(4, 4))
        self.model.swinv2.embeddings.patch_embeddings.norm = nn.LayerNorm((192,), eps=1e-05, elementwise_affine=True)
        self.model.swinv2.embeddings.norm = nn.LayerNorm((192,), eps=1e-05, elementwise_affine=True)
        self.model.swinv2.encoder.layers.pop(0)
        self.conv = nn.Conv2d(768, 384, 1)

    def forward(self, x):
        x_out = self.model(
            x, output_hidden_states=True
        ).reshaped_hidden_states[-1]  # last hidden states
        return rearrange(self.conv(x_out), "b c h w -> b (h w) c")


class TrOCRDecoder(nn.Module):
    def __init__(self, config: SwinTransformerOCRConfig) -> None:
        super().__init__()
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-small-printed"
        ).decoder
        self.model.set_input_embeddings(nn.Embedding(OCRTokenizer.length, 256))
        self.model.set_output_embeddings(nn.Linear(256, OCRTokenizer.length))

    def forward(
        self,
        input_ids: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
        attention_mask: torch.Tensor = None,
        past_key_values: torch.FloatTensor = None,
        use_cache: bool = False

    ):
        output = self.model(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        if use_cache:
            return output.logits, output.past_key_values
        else:
            return output.logits


class SwinTransformerOCR(pl.LightningModule):
    def __init__(self, config: SwinTransformerOCRConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = SwinTransformerEncoder(config)
        self.decoder = TrOCRDecoder(config)
        self.tokenizer = OCRTokenizer()
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing,
            ignore_index=self.tokenizer.pad_id
        )

        self.save_hyperparameters()

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: torch.FloatTensor = None,
        use_cache: bool = False
    ):
        encoder_hidden_states = self.encoder(images)
        return self.decoder(
            input_ids=labels,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache
        )

    def _forward_step(self, batch: tuple, stage: str = "train"):
        images, labels, labels_shifted, attn_mask_in, attn_mask_out = batch
        logits = self.forward(images, labels, attn_mask_in)
        pad_mask = rearrange(attn_mask_out, "b s -> (b s)")
        labels_shifted = rearrange(labels_shifted, "b s -> (b s)")[pad_mask == 1]
        logits = rearrange(logits, "b s c -> (b s) c")[pad_mask == 1]
        loss = self.criterion(logits, labels_shifted)

        with torch.no_grad():
            labels_pred = torch.argmax(logits.detach(), dim=-1)
            acc = torch.mean((labels_pred == labels_shifted).float())
            # print(f"{acc.item() = }, {loss.item() = }, {labels_pred.tolist() = }")
            self.log_dict({
                f"loss_{stage}": loss.detach().item(),
                f"acc_{stage}": acc
            }, sync_dist=True)
        return loss

    def training_step(self, batch: tuple, batch_idx: int):
        return self._forward_step(batch, stage="train")

    def validation_step(self, batch: tuple, batch_idx: int):
        return self._forward_step(batch, stage="val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
            eps=self.config.eps
        )
        scheduler = {
            'scheduler': OneCycleLR(
                optimizer=optimizer,
                max_lr=self.config.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=self.config.pct_start,
                div_factor=self.config.div_factor,
                final_div_factor=self.config.final_div_factor
            ),
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    @torch.no_grad()
    def predict(self, images: torch.FloatTensor):
        n_samples = images.shape[0]
        check_eos = torch.zeros(n_samples)
        labels = torch.zeros(n_samples, 1).long()  # <sos>
        list_inference = [[] for _ in range(n_samples)]
        token_count = 0
        encoder_hidden_states = self.encoder(images)
        past_key_values = None
        while (check_eos.sum().item() < n_samples) and (token_count < self.config.max_tokens):
            token_count += 1
            logits, past_key_values = self.decoder(
                input_ids=labels,
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=past_key_values,
                use_cache=True
            )
            labels = torch.argmax(logits, dim=-1, keepdim=True)
            for i in range(n_samples):
                token = labels[i][0].item()
                if token == 1:  # <eos>
                    check_eos[i] = 1
                elif (token != 1) and (check_eos[i] != 1):
                    list_inference[i].append(token)

        return {
            "output": self.tokenizer.batch_decode(list_inference),
            "output_ids": list_inference
        }
