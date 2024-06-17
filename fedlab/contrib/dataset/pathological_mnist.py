# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from .basic_dataset import FedDataset, BaseDataset
from ...utils.dataset.functional import noniid_slicing, random_slicing
from pathlib import Path

class PathologicalMNIST(FedDataset):
    """The partition stratigy in FedAvg. See http://proceedings.mlr.press/v54/mcmahan17a?ref=https://githubhelp.com

        Args:
            root (str): Path to download raw dataset.
            path (str): Path to save partitioned subdataset.
            num_clients (int): Number of clients.
            shards (int, optional): Sort the dataset by the label, and uniformly partition them into shards. Then 
            download (bool, optional): Download. Defaults to True.
        """
    def __init__(self, data_root, num_clients=100, shards=200, download=True, preprocess=False, transform=None) -> None:
        self.data_root = data_root
        self.home = Path.home()
        self.num_clients = num_clients
        self.shards = shards
        self.transform = transform
        if preprocess:
            self.preprocess(download)

    def preprocess(self, download=True):
        # self.num_clients = num_clients≠–
        # self.shards = shards
        self.download = download

        # train
        mnist = torchvision.datasets.MNIST(self.data_root, train=True, download=self.download,
                                           transform=transforms.ToTensor())
        data_indices = noniid_slicing(mnist, self.num_clients, self.shards)

        samples, labels = [], []
        for x, y in mnist:
            samples.append(x)
            labels.append(y)
        for id, indices in data_indices.items():
            data, label = [], []
            for idx in indices:
                x, y = samples[idx], labels[idx]
                data.append(x)
                label.append(y)
            dataset = BaseDataset(data, label)
            torch.save(dataset, os.path.join(self.home, 'MNIST', "train", "data{}.pkl".format(id)))

    def get_dataset(self, id, type="train"):
        """Load subdataset for client with client ID ``cid`` from local file.

        Args:
            cid (int): client id
            type (str, optional): Dataset type, can be ``"train"``, ``"val"`` or ``"test"``. Default as ``"train"``.

        Returns:
            Dataset
        """
        dataset = torch.load(os.path.join(self.home, type, "data{}.pkl".format(id)))
        return dataset

    def get_dataloader(self, id, batch_size=None, type="train"):
        """Return dataload for client with client ID ``cid``.

        Args:
            cid (int): client id
            batch_size (int, optional): batch size in DataLoader.
            type (str, optional): Dataset type, can be ``"train"``, ``"val"`` or ``"test"``. Default as ``"train"``.
        """
        dataset = self.get_dataset(id, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader
