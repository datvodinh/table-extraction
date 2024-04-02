from transformers import GPT2Tokenizer
from text_recognition.config import TransformerOCRConfig


class OCRTokenizer:
    def __init__(self, config: TransformerOCRConfig) -> None:
        self._tokenizer = GPT2Tokenizer.from_pretrained('gpt2', trust_remote_code=True)
        self._tokenizer.add_special_tokens(config.special_tokens)

    @property
    def pad_token_id(self):
        return self._tokenizer.pad_token_id

    def __len__(self):
        return self._tokenizer

    def encode(self, text: str):
        return self._tokenizer.encode(text=text)

    def batch_encode_plus(self, labels: list[str]):
        return self._tokenizer.batch_encode_plus(
            labels,
            padding=True,
            return_tensors="pt"
        )

    def decode(self, token_ids: list[int]):
        return GPT2Tokenizer.decode(token_ids=token_ids)

    def batch_decode(self, sequences: list[list[int]]):
        return GPT2Tokenizer.batch_decode(
            sequences=sequences,
            skip_special_tokens=True
        )
