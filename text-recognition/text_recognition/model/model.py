import torch
import torch.nn as nn
import pytorch_lightning as pl
from einops import rearrange
from transformers import TrOCRForCausalLM, TrOCRConfig
from text_recognition.config import SwinTransformerOCRConfig
from text_recognition.tokenizer import OCRTokenizer
# from torch.optim.lr_scheduler import OneCycleLR
from torchvision.models.swin_transformer import swin_v2_t, Swin_V2_T_Weights


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1).features

    def forward(self, images):
        return rearrange(self.encoder(images), "b h w c -> b (h w) c")


class Decoder(nn.Module):
    def __init__(self, config: SwinTransformerOCRConfig) -> None:
        super().__init__()
        self.decoder = TrOCRForCausalLM(
            TrOCRConfig(
                vocab_size=OCRTokenizer.length,
                d_model=config.d_model,
                decoder_layers=config.decoder_layers,
                decoder_attention_heads=config.decoder_attention_heads,
                decoder_ffn_dim=config.decoder_ffn_dim,
                decoder_start_token_id=OCRTokenizer.bos_id,
                pad_token_id=OCRTokenizer.pad_id,
                bos_token_id=OCRTokenizer.bos_id,
                eos_token_id=OCRTokenizer.eos_id
            )
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
        attention_mask: torch.Tensor = None,
        past_key_values: torch.FloatTensor = None,
        use_cache: bool = False

    ):
        output = self.decoder(
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
        self.encoder = Encoder()
        self.decoder = Decoder(config)
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
        images, labels, labels_shifted, attn_mask = batch
        logits = self.forward(images, labels, attn_mask)
        labels_shifted = rearrange(labels_shifted, "b s -> (b s)")
        logits = rearrange(logits, "b s c -> (b s) c")
        loss = self.criterion(logits, labels_shifted)
        with torch.no_grad():
            acc = (torch.argmax(logits, dim=-1) == labels_shifted).float().mean()
            self.log_dict({
                f"loss_{stage}": loss,
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
        )
        # scheduler = {
        #     'scheduler': OneCycleLR(
        #         optimizer=optimizer,
        #         max_lr=self.config.lr,
        #         total_steps=self.trainer.estimated_stepping_batches,
        #         pct_start=self.config.pct_start
        #     ),
        #     'interval': 'step',
        #     'frequency': 1
        # }
        return [optimizer]  # , [scheduler]

    @torch.no_grad()
    def batch_inference(self, images: torch.FloatTensor):
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
