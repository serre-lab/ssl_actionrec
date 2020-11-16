import os
import pickle
import argparse
from warnings import warn

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

from datasets.NTU import NTU_SSL, NTU_Dataset
from datasets.UCLA import UCLA_SSL, UCLA_Dataset


class FeatureDataModule(LightningDataModule):  # pragma: no cover

    name = 'Feature'

    def __init__(
            self,
            train_features,
            train_labels,
            val_features,
            val_labels,
            num_workers=4,
            batch_size=32,
            *args,
            **kwargs,
    ):
        super().__init__() # *args, **kwargs

        self.num_workers = num_workers        
        self.batch_size = batch_size

        self.train_features = train_features
        self.train_labels = train_labels

        self.val_features = val_features
        self.val_labels = val_labels


    @property
    def num_classes(self):
        return int(self.train_labels.max()) + 1
    
    def train_dataloader(self):
        
        dataset = FeatureDataset(self.train_features, self.train_labels)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
        )
        return loader

    def val_dataloader(self):
        dataset = FeatureDataset(self.val_features, self.val_labels)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader

    def test_dataloader(self):
        dataset = FeatureDataset(self.val_features, self.val_labels)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader

    def _default_transforms(self):
        transforms = transform_lib.Compose([
            tools.ToTensor()
        ])
        return transforms


class FeatureDataset(Dataset):
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            data = self.features[index]
            labels = self.labels[index]

            return data, labels 


