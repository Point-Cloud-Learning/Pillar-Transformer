import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


def normalization(one_data):
    centroid = np.mean(one_data, axis=0)
    one_data = one_data - centroid
    m = np.max(np.sqrt(np.sum(one_data ** 2, axis=1)))
    one_data = one_data / m
    return one_data


class MyDataSet(Dataset):
    def __init__(self, file_pathway, transforms=None):
        super(MyDataSet, self).__init__()
        self.path = file_pathway
        self.transforms = transforms
        self.file = h5py.File(file_pathway, 'r')
        self.data = np.array(self.file["data"])
        self.label = np.array(self.file["label"])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        one_data = self.data[item]
        one_label = self.label[item]
        if self.transforms is not None:
            one_data, one_label = self.transforms(one_data, one_label)
        return one_data, one_label


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data, label):
        for t in self.transforms:
            data, label = t(data, label)
        return data, label


class ToTensor:
    def __call__(self, data, label):
        return torch.as_tensor(data, dtype=torch.float32), torch.as_tensor(label, dtype=torch.int64)

