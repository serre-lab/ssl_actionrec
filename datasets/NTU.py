
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

from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence



class NTU_SSL(LightningDataModule):  # pragma: no cover

    name = 'NTU_SSL'

    def __init__(
            self,
            train_split,
            train_split_labels,
            train_transforms,
            val_split,
            val_split_labels,
            val_transforms,
            num_workers=4,
            drop_last=False,
            shuffle=True,
            batch_size=32,
            data_fraction=1.0,
            *args,
            **kwargs,
    ):
        super().__init__() # *args, **kwargs

        self.num_workers = num_workers

        self.data_fraction = data_fraction
        self.batch_size = batch_size

        self.train_split = train_split
        self.train_split_labels = train_split_labels
        self.train_transforms = train_transforms

        self.val_split = val_split
        self.val_split_labels = val_split_labels
        self.val_transforms = val_transforms

        self.drop_last = drop_last
        self.shuffle = shuffle
        
        dataset = NTU_Dataset(self.train_split, self.train_split_labels, data_fraction=data_fraction)
        self.train_data = dataset.data
        self.train_meta = dataset.meta

        self.num_classes_ = max(self.train_meta['labels'])+1

        dataset = NTU_Dataset(self.val_split, self.val_split_labels, data_fraction=data_fraction)
        self.val_data = dataset.data
        self.val_meta = dataset.meta

        # self.load_data()

    @property
    def num_classes(self):
        return self.num_classes_ #60
    
    def train_dataset(self, transform=''):
        
        transform = self.train_transforms if transform =='' else transform

        if transform=='none':
            transforms = None
        elif transform is None:
            transforms = self._default_transforms()
        elif isinstance(transform, str):
            transforms = get_transforms(transform)
        else:
            transforms = transform

        dataset = NTU_Dataset(self.train_split, self.train_split_labels, transforms = transforms, data=self.train_data, meta=self.train_meta, data_fraction=self.data_fraction)
        return dataset

    def val_dataset(self, transform=''):
        
        transform = self.val_transforms if transform =='' else transform

        if transform=='none':
            transforms = None
        elif transform is None:
            transforms = self._default_transforms()
        elif isinstance(transform, str):
            transforms = get_transforms(transform)
        else:
            transforms = transform

        dataset = NTU_Dataset(self.val_split, self.val_split_labels, transforms = transforms, data=self.val_data, meta=self.val_meta, data_fraction=self.data_fraction)
        return dataset

    def train_dataloader(self, batch_size=None, shuffle=True, drop_last=None, transform=''): #, num_images_per_class=-1, add_normalize=False

        transform = self.train_transforms if transform =='' else transform
        
        if transform=='none':
            transforms = None
        elif transform is None:
            transforms = self._default_transforms()
        elif isinstance(transform, str):
            transforms = get_transforms(transform)
        else:
            transforms = transform

        dataset = NTU_Dataset(self.train_split, self.train_split_labels, transforms = transforms, data=self.train_data, meta=self.train_meta, data_fraction=self.data_fraction)
        drop_last = self.drop_last if drop_last is None else drop_last
        shuffle = self.shuffle if shuffle is None else shuffle
        batch_size = self.batch_size if batch_size is None else batch_size

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

    def val_dataloader(self, batch_size=None, transform=''): # , batch_size, num_images_per_class=50, add_normalize=False
        
        transform = self.val_transforms if transform =='' else transform
        
        if transform=='none':
            transforms = None
        elif transform is None:
            transforms = self._default_transforms()
        elif isinstance(transform, str):
            transforms = get_transforms(transform)
        else:
            transforms = transform

        batch_size = self.batch_size if batch_size is None else batch_size

        dataset = NTU_Dataset(self.val_split, self.val_split_labels, transforms = transforms, data=self.val_data, meta=self.val_meta, data_fraction=self.data_fraction)

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

        return self.val_dataloader()

    def _default_transforms(self):
        transforms = transform_lib.Compose([
            # tools.Subtract(),
            tools.Center(),
            tools.ToTensor(),
        ])
        return transforms

    @staticmethod
    def add_dataset_specific_args(parent_parser):

        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--train_split', type=str, default='')
        parser.add_argument('--train_split_labels', type=str, default='')
        parser.add_argument('--train_transforms', type=str, default='')
        parser.add_argument('--val_split', type=str, default='')
        parser.add_argument('--val_split_labels', type=str, default='')
        parser.add_argument('--val_transforms', type=str, default='')
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--data_fraction', type=float, default=1.0)
        
        return parser


def get_transforms(transforms):
    return vars(sk_transforms)[transforms]()


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


class NTU_Dataset(Dataset):
    def __init__(self,
                 data_path,
                 label_path,
                 transforms=None,
                 data=None,
                 meta=None,
                 data_fraction=1.0,
                 *args, **kwargs):

        self.data_path = data_path
        self.label_path = label_path
        self.transforms = transforms if transforms is not None else lambda x:x
        self.data = data
        self.meta = meta
        self.fraction = data_fraction
        self.load_data()
        
    def load_data(self):
        # data: N C V T M
        if self.data is None:
            self.data = np.load(self.data_path)
        if self.meta is None:
            self.meta = np.load(self.label_path, allow_pickle=True).item()
        self.indices = self.meta['indices']
        self.seq_len = self.meta['seq_len']
        self.one_person = self.meta['one_person']
        self.labels = self.meta['labels']
        if self.fraction < 1.0:
            size = int(self.fraction * len(self.labels))
            # self.data = self.data[:int(self.fraction * size)]
            self.indices = self.indices[:size]
            self.seq_len = self.seq_len[:size]
            self.one_person = self.one_person[:size]
            self.labels = self.labels[:size]
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.one_person[index]==1:
            data_numpy = self.data[self.indices[index]: self.indices[index]+self.seq_len[index]]
            data_numpy = np.stack([data_numpy, np.zeros_like(data_numpy)], 0)
        else:
            data_numpy = self.data[self.indices[index]: self.indices[index]+self.seq_len[index]*2]
            data_numpy = np.stack([data_numpy[:self.seq_len[index]], data_numpy[self.seq_len[index]:]], 0)
        
        data_numpy = data_numpy.transpose([1,0,2,3])

        label = self.labels[index]

        # data, augs = self.transforms(data_numpy)
        data = self.transforms(data_numpy)
        
        # return data, augs, label 
        return data, label 
