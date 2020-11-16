
import h5py
import numpy as np

import os
import pickle
import argparse
from warnings import warn

import numpy as np

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms as transform_lib

from datasets import tools
from datasets import transforms as sk_transforms

import torch
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence

# from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
# from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence


class UCLA_SSL(LightningDataModule):  # pragma: no cover

    name = 'UCLA_SSL'

    def __init__(
            self,
            train_split,
            train_transforms,
            val_split,
            val_transforms,
            num_workers=4,
            batch_size=32,
            data_fraction=1.0,
            *args,
            **kwargs,
    ):
        super().__init__()

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_split = train_split
        self.train_transforms = train_transforms
        self.val_split = val_split
        self.val_transforms = val_transforms

        self.data_fraction = data_fraction

    @property
    def num_classes(self):
        return 10

    def train_dataset(self, transform=''):
        
        transform = self.train_transforms if transform =='' else transform
        
        if transform=='none':
            transforms = None
        elif transform is None:
            transforms = self._default_transforms()
        else:
            transforms = get_transforms(transform)
            
        dataset = UCLA_Dataset(self.train_split, transforms = transforms, data_fraction=self.data_fraction)
        return dataset

    def val_dataset(self, transform=''):
        
        transform = self.val_transforms if transform =='' else transform
        
        if transform=='none':
            transforms = None
        elif transform is None:
            transforms = self._default_transforms()
        else:
            transforms = get_transforms(transform)

        dataset = UCLA_Dataset(self.val_split, transforms = transforms, data_fraction=self.data_fraction)
        return dataset

    def train_dataloader(self, batch_size=None, shuffle=True, drop_last=True, transform=''):

        transform = self.train_transforms if transform =='' else transform
        transforms = get_transforms(transform) if transform is not None else self._default_transforms()

        batch_size = self.batch_size if batch_size is None else batch_size

        dataset = UCLA_Dataset(self.train_split, transforms = transforms, data_fraction=self.data_fraction)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=drop_last,
            pin_memory=True,
            collate_fn=pad_collate,
        )
        return loader

    def val_dataloader(self, batch_size=None, transform=''):

        transform = self.val_transforms if transform =='' else transform
        transforms = get_transforms(transform) if transform is not None else self._default_transforms()
        
        batch_size = self.batch_size if batch_size is None else batch_size

        dataset = UCLA_Dataset(self.val_split, transforms = transforms, data_fraction=self.data_fraction)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=pad_collate,
        )
        return loader

    def test_dataloader(self, transform=''):
        return self.val_dataloader(transform='')

    def _default_transforms(self):
        transforms = transform_lib.Compose([
            tools.ToTensor()
        ])
        return transforms

    @staticmethod
    def add_dataset_specific_args(parent_parser):

        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--train_split', type=str, default='')
        parser.add_argument('--train_transforms', type=str, default='')
        parser.add_argument('--val_split', type=str, default='')
        parser.add_argument('--val_transforms', type=str, default='')
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--num_workers', type=int, default=4)
        
        return parser


def get_transforms(transforms):
    return vars(sk_transforms)[transforms]()

def get_data_list(data_path):
    f = h5py.File(data_path, "r")
    data_list = []
    label_list = []

    for i in range(len(f["label"])):
        if np.shape(f[str(i)][:])[0] > 10:
            x = f[str(i)][:]
            y = f["label"][i]

            data_list.append(np.array(x))
            label_list.append(y)

    return data_list, label_list


def pad_collate(batch):
    if isinstance(batch[0][0], tuple):

        tuples = []
        for i in range(len(batch[0][0])):
            data = [x[0][i] for x in batch]
            lens = [x.shape[0] for x in data]
            xx_pad = pad_sequence(data, batch_first=True, padding_value=0)
            tuples.append( (xx_pad, lens) )

        label = [x[1] for x in batch]
        label = np.asarray(label)
        
        return tuples, label 

    else:

        data = [x[0] for x in batch]
        lens = [x.shape[0] for x in data]

        label = [x[1] for x in batch]
        label = np.asarray(label)

        xx_pad = pad_sequence(data, batch_first=True, padding_value=0)
        
        return (xx_pad, lens), label


class UCLA_Dataset(Dataset):
    def __init__(self, data_path, transforms, data_fraction=1.0):

        self.data, self.label = get_data_list(data_path)
        self.transforms = transforms
        self.data_fraction = data_fraction
        label = np.asarray(self.label)
        train_index = np.zeros(len(self.label))

    def __getitem__(self, index):
        sequence = self.data[index]
        label = self.label[index]
        
        sequence = sequence.reshape([-1,1,20,3])
        sequence = self.transforms(sequence) # .float()

        return sequence, label

    def __len__(self):
        return int(len(self.label)*self.data_fraction)
