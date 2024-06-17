import copy
import os.path

from fedlab.contrib.dataset.pathological_mnist import PathologicalMNIST
from fedlab.contrib.dataset.partitioned_cifar10 import PartitionedCIFAR10
from fedlab.contrib.dataset.partitioned_cifar100 import PartitionedCIFAR100

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision

IMAGE_SIZE = 32

def load_datasets(args, data_root, json_path, trainer):
    json_path = os.path.join(json_path, args.dataset)

    if args.dataset == 'mnist':

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        dataset = PathologicalMNIST(data_root=data_root, num_clients=args.total_client, transform=transform)

        dataset.preprocess()

        trainer.setup_dataset(dataset)
        test_data = torchvision.datasets.MNIST(root=data_root,
                                               train=False,
                                               transform=transform)
        test_loader = DataLoader(test_data, batch_size=1024)
    elif args.dataset == 'cifar10':

        if args.transform:
            train_transform = transforms.Compose([
                transforms.RandomCrop(IMAGE_SIZE, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))])

        dataset = PartitionedCIFAR10(data_root=data_root, num_clients=args.total_client, transform=train_transform, json_path=json_path, dir_alpha=args.dir_alpha)
        dataset.preprocess(balance=args.balance,
                           partition=args.partition,
                           dir_alpha=args.dir_alpha,
                           batch_size=args.batch_size,
                           )

        trainer.setup_dataset(dataset)
        test_data = torchvision.datasets.CIFAR10(root=data_root,
                                                 train=False,
                                                 transform=test_transform)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    elif args.dataset == 'cifar100':
        if args.transform:
            train_transform = transforms.Compose([
                transforms.RandomCrop(IMAGE_SIZE, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761))])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        dataset = PartitionedCIFAR100(data_root=data_root, num_clients=args.total_client, transform=train_transform, json_path=json_path, dir_alpha=args.dir_alpha)
        dataset.preprocess(balance=args.balance,
                           partition=args.partition,
                           dir_alpha=args.dir_alpha,
                           batch_size=args.batch_size,
                           )

        trainer.setup_dataset(dataset)
        test_data = torchvision.datasets.CIFAR100(root=data_root,
                                                 train=False,
                                                 transform=test_transform)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    else:
        raise ValueError(f"check args.dataset")

    return dataset, test_loader
