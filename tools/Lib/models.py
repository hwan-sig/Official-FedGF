from fedlab.models.mlp import MLP
from fedlab.models.cnn import *
from fedlab.models.FedSAMcnn import *
from fedlab.models.resnet_cifar100_del_batch import ResNet18 as resnet18_DelBatch

def get_model(args):
    if args.model == 'MLP':
        if args.dataset == 'mnist':
            model = MLP(784, 10)
    elif args.model == 'cnn':
        if args.dataset == 'mnist':
            model = CNN_MNIST()
        elif args.dataset == 'femnist':
            model = CNN_FEMNIST()
        elif args.dataset == 'cifar10':
            model = CNN_CIFAR10()
        elif args.dataset == 'cifar100':
            model = CNN_CIFAR100()
    elif args.model == 'FedSAMcnn':
        if args.dataset == 'cifar10':
            model = FedsamCNN_CIFAR10()
        elif args.dataset == 'cifar100':
            model = FedsamCNN_CIFAR100()
    elif args.model == 'resnet18_nonorm':
        if args.dataset == 'cifar100':
            model = resnet18_DelBatch()
    return model
