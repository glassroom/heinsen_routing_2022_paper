# coding: utf-8

from enum import Enum

import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as tvt

from PIL import Image
from nltk import Tree

import datasets


### Globals

class Modality(Enum):
    TEXT = "TEXT"
    IMGS = "IMGS"

DATASET_DIR = '.data'


### Dataset definitions

class SSTDataset(torch.utils.data.Dataset):
    """Dataset for SST2/SST5, with/without subtrees, cleaned up, deduplicated."""

    def __init__(self, split, with_subtrees=False, fine_grained=False, cache_dir=DATASET_DIR, revision='15c608a'):
        lines = (sample['ptb_tree'] for sample in datasets.load_dataset('sst', name='ptb', split=split, revision=revision, cache_dir=cache_dir))  # text lines in ptb format
        trees = (t for l in lines for t in Tree.fromstring(l).subtrees()) if with_subtrees else (Tree.fromstring(l) for l in lines)  # trees as nltk.Tree objects
        deduplicated = { ' '.join(t.leaves()): int(t.label()) for t in trees }  # dict maps each unique text to one label
        def _clean(text):
            return text.replace(" `` ", ' "').replace(" '' ", '" ').replace(" ` ", " '").replace(" ' ", "' ").replace(" , ", ", ").replace(" .", ".").replace("\\", "").replace(" '", "'")
        if fine_grained:
            self.data = [{ 'text': _clean(text), 'label': label } for text, label in deduplicated.items()]
            self.classes = [0, 1, 2, 3, 4]
        else:
            self.data = [{ 'text': _clean(text), 'label': int(label > 2) } for text, label in deduplicated.items() if label != 2]
            self.classes = [0, 1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class IMDBDataset(torch.utils.data.Dataset):
    """Dataset for IMDB, replacing '<br />' with '\n'."""

    def __init__(self, split, cache_dir=DATASET_DIR, revision='de29c68'):
        self.data = datasets.load_dataset('imdb', split=split, cache_dir=DATASET_DIR, revision=revision)
        self.classes = [0, 1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return { 'text': sample['text'].replace('<br />', '\n'), 'label': sample['label'], }


### Helpfer function for getting dataset splits

def get_dataset_splits(dataset_name):
    if dataset_name == 'sst5':
        splits = {
            'train': SSTDataset('train',      with_subtrees=True,  fine_grained=True),
            'valid': SSTDataset('validation', with_subtrees=False, fine_grained=True),
            'test':  SSTDataset('test',       with_subtrees=False, fine_grained=True),
            'modality': Modality.TEXT, 'n_classes': 5,
        }

    elif dataset_name == 'sst2':
        splits = {
            'train': SSTDataset('train',      with_subtrees=True,  fine_grained=False),
            'valid': SSTDataset('validation', with_subtrees=False, fine_grained=False),
            'test':  SSTDataset('test',       with_subtrees=False, fine_grained=False),
            'modality': Modality.TEXT, 'n_classes': 2,
        }

    elif dataset_name == 'imdb':
        splits = {
            'train': IMDBDataset('train'),
            'valid': IMDBDataset('test'),
            'modality': Modality.TEXT, 'n_classes': 2,
        }
        splits['test'] = splits['valid']  # dataset has only 2 splits

    elif dataset_name == 'imagenet-1k':
        d, crop, interp, mean, std = (224, 0.9, 3, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # preprocessing for beit 224 patch 16 model
        _resize = tvt.Resize((int(d / crop), int(d / crop)), interpolation=interp)  # resize to square image of uncropped dim
        trn_tfm = tvt.Compose([_resize, tvt.RandomCrop(d), tvt.RandAugment(), tvt.ToTensor(), tvt.Normalize(mean, std=std)])
        tst_tfm = tvt.Compose([_resize, tvt.CenterCrop(d),                    tvt.ToTensor(), tvt.Normalize(mean, std=std)])
        splits = {
            'train': tv.datasets.ImageNet(root=f'{DATASET_DIR}/vision/imagenet/', split='train', transform=trn_tfm),
            'valid': tv.datasets.ImageNet(root=f'{DATASET_DIR}/vision/imagenet/', split='val',   transform=tst_tfm),
            'modality': Modality.IMGS, 'n_classes': 1000,
        }
        splits['test'] = splits['valid']  # convention is to report accuracy on val split

    elif dataset_name == 'cifar100':
        d, crop, interp, mean, std = (224, 0.9, 3, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # preprocessing for beit 224 patch 16 model
        trn_tfm = tvt.Compose([tvt.Resize(d, interpolation=interp), tvt.RandAugment(), tvt.ToTensor(), tvt.Normalize(mean, std=std)])
        tst_tfm = tvt.Compose([tvt.Resize(d, interpolation=interp),                    tvt.ToTensor(), tvt.Normalize(mean, std=std)])
        splits = {
            'train': tv.datasets.CIFAR100(root=f'{DATASET_DIR}/vision/cifar100/train/', train=True,  transform=trn_tfm, download=True),
            'valid': tv.datasets.CIFAR100(root=f'{DATASET_DIR}/vision/cifar100/test/',  train=False, transform=tst_tfm, download=True),
            'modality': Modality.IMGS, 'n_classes': 100,
        }
        splits['test'] = splits['valid']  # dataset has only 2 splits

    elif dataset_name == 'cifar10':
        d, crop, interp, mean, std = (224, 0.9, 3, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # preprocessing for beit 224 patch 16 model
        trn_tfm = tvt.Compose([tvt.Resize(d, interpolation=interp), tvt.RandAugment(), tvt.ToTensor(), tvt.Normalize(mean, std=std)])
        tst_tfm = tvt.Compose([tvt.Resize(d, interpolation=interp),                    tvt.ToTensor(), tvt.Normalize(mean, std=std)])
        splits = {
            'train': tv.datasets.CIFAR10(root=f'{DATASET_DIR}/vision/cifar10/train/', train=True,  transform=trn_tfm, download=True),
            'valid': tv.datasets.CIFAR10(root=f'{DATASET_DIR}/vision/cifar10/test/',  train=False, transform=tst_tfm, download=True),
            'modality': Modality.IMGS, 'n_classes': 10,
        }
        splits['test'] = splits['valid']  # dataset has only 2 splits
    else:
        raise ValueError(f'Unrecognized dataset name: "{dataset_name}".')

    return splits