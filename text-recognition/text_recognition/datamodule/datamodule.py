
import os
import random
import torch
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from functools import partial
from text_recognition.config import TransformerOCRConfig
from text_recognition.datamodule.transform import OCRTransform
from transformers import GPT2Tokenizer


class OCRDataset(Dataset):
    def __init__(
        self,
        config: TransformerOCRConfig,
        list_path: str,
        stage: str = "train"
    ):
        self.config = config
        self.list_path = list_path
        self.stage = stage
        self.img_size = config.img_size
        self.transform = OCRTransform(config.img_size, stage)

    def __len__(self):
        return len(self.list_path)

    def __getitem__(self, index):
        img = Image.open(self.list_path[index][0])
        with open(self.list_path[index][1], encoding="utf-8") as f:
            label = f.read()
        return self.transform(image=img), "<|SEP|>" + label + "<|endoftext|>"


class OCRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: TransformerOCRConfig,
        data_dir: str
    ):
        super().__init__()
        self.in_channels = config.in_channels
        self.config = config
        self.data_dir = data_dir

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', trust_remote_code=True)
        self.loader = partial(
            DataLoader,
            collate_fn=self.collate_fn,
            batch_size=config.batch_size,
            pin_memory=True,
            num_workers=config.num_workers,
            persistent_workers=True
        )

    def collate_fn(self, batch):
        images, labels = batch
        images = torch.stack(images)
        data = self.tokenizer.batch_encode_plus(
            labels,
            padding=True,
            return_tensors="pt"
        )
        attn_mask = data['attention_mask']
        input_ids = data['input_ids'][:, :-1]
        input_ids_shifted = data['input_ids'][:, 1:]
        return (images, input_ids, input_ids_shifted, attn_mask)

    def setup(self, stage: str):
        if stage == "fit":
            list_path = []
            for folder, _, files in os.walk(self.data_dir):
                for name in files:
                    for ftype in [".png", ".jpg", ".jpeg"]:
                        if ftype in name:
                            image_path = os.path.join(folder, name)
                            label_path = os.path.join(folder, name.replace(ftype, ".txt"))
                            if os.path.isfile(label_path):
                                list_path.append([image_path, label_path])
            random.shuffle(list_path)
            len_train = int(len(list_path) * self.config.train_ratio)
            train_list = list_path[:len_train]
            val_list = list_path[len_train:]
            self.OCR_train = OCRDataset(
                config=self.config, list_path=train_list, stage="train"
            )
            self.OCR_val = OCRDataset(
                config=self.config, list_path=val_list, stage="val"
            )
        else:
            pass

    def train_dataloader(self):
        return self.loader(dataset=self.OCR_train)

    def val_dataloader(self):
        return self.loader(dataset=self.OCR_val)
