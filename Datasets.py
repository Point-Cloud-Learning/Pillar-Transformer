import os
import h5py
import numpy as np
import torch

from torch.utils.data import Dataset


def normalization(one_data):
    centroid = np.mean(one_data, axis=0)
    one_data = one_data - centroid
    m = np.max(np.sqrt(np.sum(one_data ** 2, axis=1)))
    one_data = one_data / m
    return one_data


class ModelNet(Dataset):
    def __init__(self, root="./data/modelnet40_normal_resampled/", num_point=1024, split="train", uniform_sampling="directly", use_normals=False):
        super(ModelNet, self).__init__()
        self.root = root
        self.num_point = num_point
        self.use_normals = use_normals
        self.uniform_sampling = uniform_sampling

        self.cat_file = os.path.join(self.root, "modelnet40_shape_names.txt")
        self.cat = [line.rstrip() for line in open(self.cat_file)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {"train": [line.rstrip() for line in open(os.path.join(self.root, "modelnet40_train.txt"))],
                     "test": [line.rstrip() for line in open(os.path.join(self.root, "modelnet40_test.txt"))]}

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.data_path = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i in range(len(shape_ids[split]))]

        self.cache = {}

        # label weights
        self.num_labels = np.zeros(len(self.classes))
        for cls in self.data_path:
            self.num_labels[self.classes[cls[0]]] += 1
        self.num_labels = self.num_labels.astype(np.float32)
        label_weights = self.num_labels / np.sum(self.num_labels)
        self.label_weights = torch.from_numpy(np.power(np.amax(label_weights) / label_weights, 1 / 3.0))

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, item):
        if item in self.cache:
            point_set, cls = self.cache[item]
        else:
            fn = self.data_path[item]
            cls = self.classes[fn[0]]
            cls = np.array(cls).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            point_set[:, 0:3] = normalization(point_set[:, 0:3])

            if self.uniform_sampling == "random&saving":
                choice = np.random.choice(len(point_set), self.num_point, replace=False)
                point_set = point_set[choice, :]
                if not self.use_normals:
                    point_set = point_set[:, 0:3]
                self.cache[item] = (point_set, cls)

            elif self.uniform_sampling == "random":
                self.cache[item] = (point_set, cls)

            else:
                point_set = point_set[0:self.num_point, :]
                if not self.use_normals:
                    point_set = point_set[:, 0:3]
                self.cache[item] = (point_set, cls)

        if self.uniform_sampling == "random":
            choice = np.random.choice(len(point_set), self.num_point, replace=False)
            point_set = point_set[choice, :]
            if not self.use_normals:
                point_set = point_set[:, 0:3]

        return point_set, cls


class Datasets(Dataset):
    def __init__(self, file_pathway, transforms=None):
        super(Datasets, self).__init__()
        self.path = file_pathway
        self.transforms = transforms
        self.file = h5py.File(file_pathway, 'r')
        self.data = np.array(self.file["data"])
        self.label = np.array(self.file["label"])

        # label weights
        self.num_labels = np.bincount(self.label.reshape((-1))).astype(np.float32)
        self.label_weights = torch.from_numpy(np.power(np.amax(self.num_labels) / self.num_labels, 1 / 3.0))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        one_data = self.data[item]
        one_label = self.label[item][0]
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

