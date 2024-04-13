import torch


class OCRTokenizer:
    vocab: str = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~ '
    length: int = len(vocab)+3
    bos_id: int = 0
    eos_id: int = 1
    pad_id: int = 2

    def __init__(self) -> None:

        self.letter_to_idx = {char: i+3 for i, char in enumerate(sorted(self.vocab))}
        self.idx_to_letter = {i+3: char for i, char in enumerate(sorted(self.vocab))}
        self.letter_to_idx['<sos>'] = 0
        self.letter_to_idx['<eos>'] = 1
        self.letter_to_idx['<pad>'] = 2
        self.idx_to_letter[0] = '<sos>'
        self.idx_to_letter[1] = '<eos>'
        self.idx_to_letter[2] = '<pad>'

    def __len__(self):
        return self.length

    def encode(self, text: str):
        indices = [self.letter_to_idx.get(i, self.pad_id) for i in text]
        return torch.tensor([self.letter_to_idx['<sos>']] + indices + [self.letter_to_idx['<eos>']])

    def batch_encode(self, labels: list[str]):
        list_indices = [
            self.encode(label)
            for label in labels
        ]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            list_indices,
            batch_first=True,
            padding_value=self.pad_id
        )
        attention_mask = (input_ids != self.pad_id).float()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

    def decode(self, token_ids: list[int]):
        chars = [self.idx_to_letter[int(i)] for i in token_ids]
        decoded_chars = [c for c in chars if c not in ['<sos>', '<eos>', '<pad>']]
        return "".join(decoded_chars)

    def batch_decode(self, sequences: list[list[int]]):
        return [
            self.decode(seq)
            for seq in sequences
        ]


if __name__ == "__main__":
    tok = OCRTokenizer()
    s = "Xin chào bạn khỏe không?"
    enc = tok.encode(s)
    dec = tok.decode(enc)
    print(enc)
    print(dec)
