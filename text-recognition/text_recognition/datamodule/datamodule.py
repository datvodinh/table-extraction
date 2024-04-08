import os
import cv2
import random
import torch
import imagesize
import multiprocessing
import pytorch_lightning as pl
import torch.distributed
from torch.utils.data import DataLoader, Dataset, BatchSampler
from text_recognition.config import SwinTransformerOCRConfig
from text_recognition.datamodule.transform import OCRTransform
from text_recognition.tokenizer import OCRTokenizer
from concurrent.futures import ThreadPoolExecutor


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
        self.max_ratio = self.config.img_size[1] / self.config.img_size[0]
        self._cluster_image_by_ratios()

    def _cluster_image_by_ratios(self):
        self.ratio_cluster = {}
        max_ratio = self.config.img_size[1] / self.config.img_size[0]

        def process_file(f):
            w, h = imagesize.get(f[0])
            r = max(round(w/h*2), 1)
            r = min(r/2, max_ratio)
            return str(r), f
        print("Getting image clusters!")
        cpu_count = multiprocessing.cpu_count()
        with ThreadPoolExecutor(max_workers=cpu_count*5 if cpu_count < 10 else cpu_count*2) as executor:
            results = list(executor.map(process_file, self.list_path))

        for r, f in results:
            if r not in self.ratio_cluster.keys():
                self.ratio_cluster[r] = [f]
            else:
                self.ratio_cluster[r].append(f)
        # for k in self.ratio_cluster.keys():
        #     print(k, len(self.ratio_cluster[k]))

    def __len__(self):
        return len(self.list_path)

    def __getitem__(self, info):
        if isinstance(info, int):
            idx = info
        else:
            idx, ratio = info
        img = cv2.imread(self.list_path[idx][0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.list_path[idx][1]
        return self.transform(image=img, ratio=ratio), label


class OCRImageClusterSampler(BatchSampler):
    def __init__(
        self,
        dataset: OCRDataset,
        batch_size: int = 32,
        shuffle: bool = True
    ) -> None:
        super().__init__(dataset, batch_size=batch_size, drop_last=False)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        list_key = list(self.dataset.ratio_cluster.keys())
        if self.shuffle:
            random.shuffle(list_key)
        list_batch_indices = []
        remain_indices = []
        remain_max_ratio = 0
        for k in list_key:
            list_path = self.dataset.ratio_cluster[k]
            if self.shuffle:
                random.shuffle(list_path)
            list_label = [lp[1] for lp in list_path]
            for i in range(0, len(list_path), self.batch_size):
                start, end = i, min(i + self.batch_size, len(list_path))
                if end == i+self.batch_size:
                    list_batch_indices.append([
                        (int(self.dataset.list_path_index[label]), float(k))
                        for label in list_label[start:end]
                    ])
                else:
                    if remain_max_ratio < float(k):
                        remain_max_ratio = float(k)
                    remain_indices += list_label[start:end]
        for i in range(0, len(remain_indices), self.batch_size):
            list_batch_indices.append([
                (int(self.dataset.list_path_index[label]), int(remain_max_ratio))
                for label in remain_indices[i:i + self.batch_size]
            ])

        if self.shuffle:
            random.shuffle(list_batch_indices)
        for batch_indices in list_batch_indices:
            yield batch_indices


class Collator:
    def __init__(self) -> None:
        self.tokenizer = OCRTokenizer()

    def __call__(self, batch):
        images = torch.stack([b[0] for b in batch])
        labels = [b[1] for b in batch]
        data = self.tokenizer.batch_encode(labels)
        attn_mask = data['attention_mask'][:, :-1]
        input_ids = data['input_ids'][:, :-1]
        input_ids_shifted = data['input_ids'][:, 1:]
        return (images, input_ids, input_ids_shifted, attn_mask)


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
        self.train_list = list_data[:len_train]
        self.val_list = list_data[len_train:]

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
            batch_sampler=OCRImageClusterSampler(
                dataset=self.OCR_train,
                batch_size=self.config.batch_size,
                shuffle=True
            ),
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
            batch_sampler=OCRImageClusterSampler(
                dataset=self.OCR_val,
                batch_size=self.config.batch_size,
                shuffle=False
            ),
            collate_fn=Collator(),
            pin_memory=True if self.config.num_workers > 0 else False,
            persistent_workers=True if self.config.num_workers > 0 else False,
            multiprocessing_context='fork' if (
                torch.backends.mps.is_available() and self.config.num_workers > 0
            ) else None
        )
