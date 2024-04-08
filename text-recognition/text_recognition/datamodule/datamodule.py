import os
import cv2
import random
import torch
import pytorch_lightning as pl
import torch.distributed
from torch.utils.data import DataLoader, Dataset
from text_recognition.config import SwinTransformerOCRConfig
from text_recognition.datamodule.transform import OCRTransform
from text_recognition.tokenizer import OCRTokenizer


class OCRDataset(Dataset):
    def __init__(
        self,
        config: SwinTransformerOCRConfig,
        list_path: list,
        stage: str = "train"
    ):
        super().__init__()
        self.config = config
        self.list_path = list_path
        self.list_path_index = {lp[1]: i for i, lp in enumerate(self.list_path)}  # ["example.txt": 1]
        self.stage = stage
        self.img_size = config.img_size
        self.transform = OCRTransform(config.img_size, stage)

    def __len__(self):
        return len(self.list_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.list_path[idx][0])
        label = self.list_path[idx][1]
        return self.transform(image=img), label


class Collator:
    def __init__(self) -> None:
        self.tokenizer = OCRTokenizer()

    def __call__(self, batch):
        images = torch.stack([b[0] for b in batch])
        labels = [b[1] for b in batch]
        data = self.tokenizer.batch_encode(labels)
        attn_mask_in = data['attention_mask'][:, :-1]
        attn_mask_out = data['attention_mask'][:, 1:]
        input_ids = data['input_ids'][:, :-1]
        input_ids_shifted = data['input_ids'][:, 1:]
        return (images, input_ids, input_ids_shifted, attn_mask_in, attn_mask_out)


class OCRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: SwinTransformerOCRConfig,
        data_dir: str
    ):
        super().__init__()
        self.in_channels = config.in_channels
        self.config = config
        self.data_dir = data_dir

        self._get_list_path()

    def _get_list_path(self):
        list_data = []
        for folder, _, files in os.walk(self.data_dir):
            label_file = os.path.join(folder, "labels.txt")
            if os.path.exists(label_file):
                with open(label_file, "r", encoding="utf-8") as f:
                    data = f.read().strip().split("\n")
                for d in data:
                    split_index = d.find('.jpg')
                    image_file = d[:split_index+4].strip()
                    label = d[split_index + 5:].strip()
                    list_data.append(
                        [
                            os.path.join(folder, image_file),
                            label
                        ]
                    )
        print(f"Total Image: {len(list_data)}")
        random.shuffle(list_data)
        len_train = int(len(list_data) * self.config.train_ratio)
        len_val = int(len(list_data) * min(self.config.train_ratio, 0.01))
        self.train_list = list_data[:len_train]
        self.val_list = list_data[len_train:len_train+len_val]

    def setup(self, stage: str = "train"):
        self.OCR_train = OCRDataset(
            config=self.config, list_path=self.train_list, stage="train"
        )
        self.OCR_val = OCRDataset(
            config=self.config, list_path=self.val_list, stage="val"
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            num_workers=self.config.num_workers,
            dataset=self.OCR_train,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=Collator(),
            pin_memory=True if self.config.num_workers > 0 else False,
            persistent_workers=True if self.config.num_workers > 0 else False,
            multiprocessing_context='fork' if (
                torch.backends.mps.is_available() and self.config.num_workers > 0
            ) else None
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            num_workers=self.config.num_workers,
            dataset=self.OCR_val,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=Collator(),
            pin_memory=True if self.config.num_workers > 0 else False,
            persistent_workers=True if self.config.num_workers > 0 else False,
            multiprocessing_context='fork' if (
                torch.backends.mps.is_available() and self.config.num_workers > 0
            ) else None
        )
