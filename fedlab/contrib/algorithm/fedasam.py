import torch

from .basic_server import SyncServerHandler
from .basic_client import SGDClientTrainer, SGDSerialClientTrainer
from .fedavg import FedAvgServerHandler
from .minimizers import ASAM
from ...utils import Aggregators

##################
#
#      Server
#
##################


class FedASamServerHandler(FedAvgServerHandler):
    pass

##################
#
#      Client
#
##################


class FedASamSerialClientTrainer(SGDSerialClientTrainer):
    def __init__(self, model, num_clients, rho, eta, cuda=True, device=None, logger=None, personal=False) -> None:
        super().__init__(model, num_clients, cuda, device, logger, personal)
        self.rho = rho
        self.eta = eta

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        for id in id_list:
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            # optimizer = torch.optim.SGD(model_parameters, lr=self.lr)
            minimizer = ASAM(self.optimizer, self.model, self.rho, self.eta)
            pack = self.train(id, model_parameters, minimizer, data_loader)
            self.cache.append(pack)

    def train(self, id, model_parameters, minimizer, train_loader):
        self.set_model(model_parameters)

        data_size = 0
        for _ in range(self.epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                # Ascent Step
                output = self.model(data)
                loss = self.criterion(output, target)

                loss.backward()
                minimizer.ascent_step()

                # Descent Step
                self.criterion(self.model(data), target).backward()
                minimizer.descent_step()

                data_size += len(target)

        return [self.model_parameters, data_size]
