import os
import cv2
import random
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from text_recognition.config import TransformerOCRConfig
from text_recognition.datamodule.transform import OCRTransform
from text_recognition.tokenizer import OCRTokenizer
from tqdm import tqdm


class OCRDataset(Dataset):
    def __init__(
        self,
        config: TransformerOCRConfig,
        list_path: list,
        stage: str = "train"
    ):
        super().__init__()
        self.config = config
        self.list_path = list_path
        self.stage = stage
        self.img_size = config.img_size
        self.transform = OCRTransform(config.img_size, stage)

    def __len__(self):
        return len(self.list_path)

    def __getitem__(self, index):
        img = cv2.imread(self.list_path[index][0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
        self.tokenizer = OCRTokenizer(config)

        self.get_list_path()

    def get_list_path(self):
        list_path = []
        for folder, _, files in tqdm(os.walk(self.data_dir)):
            for name in tqdm(files):
                for ftype in [".png", ".jpg", ".jpeg"]:
                    if ftype in name:
                        image_path = os.path.join(folder, name)
                        label_path = os.path.join(folder, name.replace(ftype, ".txt"))
                        if os.path.isfile(label_path):
                            list_path.append([image_path, label_path])
        print(f"Total Image: {len(list_path)}")
        random.shuffle(list_path)
        len_train = int(len(list_path) * self.config.train_ratio)
        self.train_list = list_path[:len_train]
        self.val_list = list_path[len_train:]

    def collate_fn(self, batch):
        images = torch.stack([b[0] for b in batch])
        labels = [b[1] for b in batch]
        data = self.tokenizer.batch_encode_plus(labels)
        attn_mask = data['attention_mask'][:, :-1]
        input_ids = data['input_ids'][:, :-1]
        input_ids_shifted = data['input_ids'][:, 1:]
        return (images, input_ids, input_ids_shifted, attn_mask)

    def setup(self, stage: str):
        self.OCR_train = OCRDataset(
            config=self.config, list_path=self.train_list, stage="train"
        )
        self.OCR_val = OCRDataset(
            config=self.config, list_path=self.val_list, stage="val"
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.OCR_train,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
            # pin_memory=True,
            # persistent_workers=True,
            # multiprocessing_context='fork' if torch.backends.mps.is_available() else None
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.OCR_val,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
            # pin_memory=True,
            # persistent_workers=True,
            # multiprocessing_context='fork' if torch.backends.mps.is_available() else None
        )
