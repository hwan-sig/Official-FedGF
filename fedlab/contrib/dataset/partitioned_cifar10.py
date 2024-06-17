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
import torchvision.transforms as transforms
import torchvision

from .basic_dataset import FedDataset, BaseDataset
from ...utils.dataset.partition import CIFAR10Partitioner
import pandas as pd
from pathlib import Path

# JSON_PATH = "/home/Hwan/FedLab/fedlab/contrib/json_data/cifar10/federated_train_alpha_"


class PartitionedCIFAR10(FedDataset):
    """:class:`FedDataset` with partitioning preprocess. For detailed partitioning, please
    check `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.

    
    Args:
        root (str): Path to download raw dataset.
        path (str): Path to save partitioned subdataset.
        dataname (str): "cifar10" or "cifar100"
        num_clients (int): Number of clients.
        download (bool): Whether to download the raw dataset.
        preprocess (bool): Whether to preprocess the dataset.
        balance (bool, optional): Balanced partition over all clients or not. Default as ``True``.
        partition (str, optional): Partition type, only ``"iid"``, ``shards``, ``"dirichlet"`` are supported. Default as ``"iid"``.
        unbalance_sgm (float, optional): Log-normal distribution variance for unbalanced data partition over clients. Default as ``0`` for balanced partition.
        num_shards (int, optional): Number of shards in non-iid ``"shards"`` partition. Only works if ``partition="shards"``. Default as ``None``.
        dir_alpha (float, optional): Dirichlet distribution parameter for non-iid partition. Only works if ``partition="dirichlet"``. Default as ``None``.
        verbose (bool, optional): Whether to print partition process. Default as ``True``.
        seed (int, optional): Random seed. Default as ``None``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """
    def __init__(self,
                 data_root,
                 # path,
                 # dataname,
                 num_clients,
                 download=True,
                 preprocess=False,
                 balance=True,
                 partition="iid",
                 unbalance_sgm=0,
                 num_shards=None,
                 dir_alpha=None,
                 json_path=None,
                 verbose=True,
                 seed=None,
                 transform=None,
                 target_transform=None) -> None:

        self.data_root = data_root
        self.home = Path.home()
        self.json_path = json_path

        self.num_clients = num_clients
        self.dir_alpha = dir_alpha
        if transform == None:
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.491, 0.482, 0.446),
                                      (0.247, 0.243, 0.261))])
        else:
            self.transform = transform

        self.targt_transform = target_transform

        if preprocess:
            self.preprocess(balance=balance,
                            partition=partition,
                            unbalance_sgm=unbalance_sgm,
                            num_shards=num_shards,
                            dir_alpha=dir_alpha,
                            verbose=verbose,
                            seed=seed,
                            download=download)

    def preprocess(self,
                   balance=True,
                   partition="iid",
                   unbalance_sgm=0,
                   num_shards=None,
                   dir_alpha=None,
                   verbose=True,
                   seed=None,
                   download=True,
                   batch_size=64):
        """Perform FL partition on the dataset, and save each subset for each client into ``data{cid}.pkl`` file.

        For details of partition schemes, please check `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.
        """
        self.download = download

        trainset = torchvision.datasets.CIFAR10(root=self.data_root,
                                                train=True,
                                                download=self.download)

        samples, labels = [], []
        for x, y in trainset:
            samples.append(x)
            labels.append(y)

        original_dataset = BaseDataset(samples, labels, self.transform)
        self.origin_data_loader = DataLoader(original_dataset, batch_size=1024, shuffle=False)

        if self.json_path:
            alpha = f'{float(dir_alpha):.2f}'
            df = pd.read_csv(os.path.join(self.json_path, 'federated_train_alpha_'+ alpha + '.csv'))
            df = df.sort_values(by=['image_id'])

            self.dataloaders = []
            dict_data_idx = dict()

            for u_idx, img, img_cls, x, y in zip(df['user_id'].values, df['image_id'].values, df['class'], samples, labels):
                assert img_cls == y, "label is not mapped!"

                if u_idx not in dict_data_idx:
                    dict_data_idx[u_idx] = {'x': [x], 'y': [y]}
                else:
                    dict_data_idx[u_idx]['x'].append(x)
                    dict_data_idx[u_idx]['y'].append(y)

            for cid in range(100):
                dataset = dict_data_idx[cid]
                B_dataset = BaseDataset(dataset['x'], dataset['y'], self.transform)
                data_loader = DataLoader(B_dataset, batch_size=batch_size, shuffle=True)#, num_workers=4)

                self.dataloaders.append(data_loader)
            return

    def get_dataset(self, cid, type="train"):
        """Load subdataset for client with client ID ``cid`` from local file.

        Args:
             cid (int): client id
             type (str, optional): Dataset type, can be ``"train"``, ``"val"`` or ``"test"``. Default as ``"train"``.

        Returns:
            Dataset
        """
        raise NotImplementedError()

    def get_dataloader(self, cid, batch_size=None, type="train"):
        """Return dataload for client with client ID ``cid``.

        Args:
            cid (int): client id
            batch_size (int, optional): batch size in DataLoader.
            type (str, optional): Dataset type, can be ``"train"``, ``"val"`` or ``"test"``. Default as ``"train"``.
        """
        return self.dataloaders[cid]
