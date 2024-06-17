import sys

from fedlab.contrib.algorithm.fedavgm import FedAvgMServerHandler
from fedlab.contrib.algorithm.fedprox import FedProxServerHandler, FedProxSerialClientTrainer
from fedlab.contrib.algorithm.scaffold import ScaffoldSerialClientTrainer, ScaffoldServerHandler
from fedlab.contrib.algorithm.feddyn import FedDynSerialClientTrainer, FedDynServerHandler

from fedlab.contrib.algorithm.fedsam import FedSamSerialClientTrainer, FedSamServerHandler
from fedlab.contrib.algorithm.fedasam import FedASamServerHandler, FedASamSerialClientTrainer
from fedlab.contrib.algorithm.mofedsam import MoFedSamServerHandler, MoFedSamSerialClientTrainer

from fedlab.contrib.algorithm.fedgf import FedGfSerialClientTrainer, FedGfServerHandler

from fedlab.contrib.algorithm.fedSMOO_woReg import FedSMOONoRegSerialClientTrainer, FedSMOONoRegServerHandler
from fedlab.contrib.algorithm.fedGamma import FedGammaServerHandler, FedGammaSerialClientTrainer

from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer
from fedlab.contrib.algorithm.fedavg import FedAvgSerialClientTrainer

def load_algorithms(args, model):
    if args.alg == "fedavg":
        handler = SyncServerHandler(model=model,
                                    global_round=args.com_round,
                                    num_clients=args.total_client,
                                    sample_ratio=args.sample_ratio,
                                    cuda=True,
                                    device='cuda:0')
        trainer = SGDSerialClientTrainer(model, args.total_client, cuda=True, device='cuda:0')
        trainer.setup_optim(args.epochs, args.batch_size, args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.alg == "fedavgm":
        handler = FedAvgMServerHandler(model=model,
                                    global_round=args.com_round,
                                    num_clients=args.total_client,
                                    sample_ratio=args.sample_ratio,
                                    cuda=True,
                                    device='cuda:0')
        handler.setup_optim(args.beta)
        trainer = FedAvgSerialClientTrainer(model, args.total_client, cuda=True, device='cuda:0')
        trainer.setup_optim(args.epochs, args.batch_size, args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.alg == "fedprox":
        handler = FedProxServerHandler(model=model,
                                       global_round=args.com_round,
                                       num_clients=args.total_client,
                                       sample_ratio=args.sample_ratio,
                                       cuda=True,
                                       device='cuda:0')
        trainer = FedProxSerialClientTrainer(model, args.total_client, cuda=True, device='cuda:0')
        trainer.setup_optim(args.epochs, args.batch_size, args.lr, weight_decay=args.weight_decay, momentum=args.momentum, mu=args.mu)
    elif args.alg == "scaffold":
        assert args.g_lr is not None
        if args.g_lr == 0:
            raise ValueError(f"{args.g_lr} must be none zero")

        handler = ScaffoldServerHandler(model=model,
                                        global_round=args.com_round,
                                        num_clients=args.total_client,
                                        sample_ratio=args.sample_ratio,
                                        cuda=True,
                                        device='cuda:0')
        handler.setup_optim(lr=args.g_lr)

        trainer = ScaffoldSerialClientTrainer(model, args.total_client, cuda=True, device='cuda:0')
        trainer.setup_optim(args.epochs, args.batch_size, args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.alg == "fedsam":
        handler = FedSamServerHandler(model=model,
                                      global_round=args.com_round,
                                      num_clients=args.total_client,
                                      sample_ratio=args.sample_ratio,
                                      cuda=True,
                                      device='cuda:0')
        trainer = FedSamSerialClientTrainer(model, args.total_client, rho=args.rho, cuda=True, device='cuda:0')
        trainer.setup_optim(args.epochs, args.batch_size, args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.alg == "fedasam":
        handler = FedASamServerHandler(model=model,
                                       global_round=args.com_round,
                                       num_clients=args.total_client,
                                       sample_ratio=args.sample_ratio,
                                       cuda=True,
                                       device='cuda:0')
        trainer = FedASamSerialClientTrainer(model, args.total_client, rho=args.rho, eta=args.eta, cuda=True, device='cuda:0')
        trainer.setup_optim(args.epochs, args.batch_size, args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.alg == "mofedsam":
        handler = MoFedSamServerHandler(model=model,
                                        global_round=args.com_round,
                                        num_clients=args.total_client,
                                        sample_ratio=args.sample_ratio,
                                        cuda=True,
                                        device='cuda:0')
        handler.setup_optim(eta_l=args.lr, eta_g=args.eta_g)

        trainer = MoFedSamSerialClientTrainer(model, args.total_client, rho=args.rho, beta=args.beta, cuda=True, device='cuda:0')
        trainer.setup_optim(args.epochs, args.batch_size, args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.alg == "fedgf":
        if args.g_rho is None:
            args.g_rho = args.rho
        handler = FedGfServerHandler(model=model,
                                     global_round=args.com_round,
                                     num_clients=args.total_client,
                                     sample_ratio=args.sample_ratio,
                                     cuda=True,
                                     device='cuda:0')
        handler.setup_optim(args.g_rho, args.T_D, args.W)

        trainer = FedGfSerialClientTrainer(model, args.total_client, rho=args.rho, cuda=True, device='cuda:0')#, use_trainable=args.use_trainable)
        trainer.setup_optim(args.epochs, args.batch_size, args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.alg == "fedsmoo_noreg":
        handler = FedSMOONoRegServerHandler(model=model,
                                            global_round=args.com_round,
                                            num_clients=args.total_client,
                                            sample_ratio=args.sample_ratio,
                                            cuda=True,
                                            device='cuda:0')
        handler.setup_optim(args.rho)

        trainer = FedSMOONoRegSerialClientTrainer(model, args.total_client, rho=args.rho, cuda=True, device='cuda:0')
        trainer.setup_optim(args.epochs, args.batch_size, args.lr, weight_decay=args.weight_decay,
                            momentum=args.momentum)
    elif args.alg == "fedgamma":
        handler = FedGammaServerHandler(model=model,
                                     global_round=args.com_round,
                                     num_clients=args.total_client,
                                     sample_ratio=args.sample_ratio,
                                     cuda=True,
                                     device='cuda:0')
        handler.setup_optim()

        trainer = FedGammaSerialClientTrainer(model, args.total_client, rho=args.rho, cuda=True, device='cuda:0')
        trainer.setup_optim(args.epochs, args.batch_size, args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.alg == "feddyn":
        handler = FedDynServerHandler(model=model,
                                      global_round=args.com_round,
                                      num_clients=args.total_client,
                                      sample_ratio=args.sample_ratio,
                                      cuda=True,
                                      device='cuda:0')
        handler.setup_optim(alpha=args.alpha)
        trainer = FedDynSerialClientTrainer(model, args.total_client, cuda=True, device='cuda:0')
        trainer.setup_optim(args.epochs, args.batch_size, args.lr, weight_decay=args.weight_decay, momentum=args.momentum, alpha=args.alpha)
    else:
        raise ValueError("check the algorithm")
    return handler, trainer


def average(lst):
    return sum(lst) / len(lst)
