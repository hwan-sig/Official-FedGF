import torch
import numpy as np
import copy
from .basic_server import SyncServerHandler
from .basic_client import SGDClientTrainer, SGDSerialClientTrainer
from .fedavg import FedAvgServerHandler
from .minimizers import MoSAM
from ...utils import Aggregators
from ...utils.serialization import SerializationTool
# batchnorm 에 대한 term 은 momentum으로 안바꾸도록.
##################
#
#      Server
#
##################


# class FedSAMServerHandler(SyncServerHandler):
class MoFedSamServerHandler(FedAvgServerHandler):
    @property
    def downlink_package(self):
        return [self.model_parameters, self.delta]

    def setup_optim(self, eta_l, eta_g=1):
        self.delta = torch.zeros_like(self.model_parameters)
        self.eta_l = eta_l
        self.eta_g = eta_g
        # self.K = K

    def global_update(self, buffer):
        self.delta = self.calc_momentum(buffer)  # grad = theta_prev - theta_current
        # super().global_update(buffer)
        serialized_parameters = self.model_parameters - self.delta*self.eta_g
        self.set_model(serialized_parameters)
        # self.set_momentum()
        # parameters_list = [ele[0] for ele in buffer]
        # weights = torch.tensor([ele[1] for ele in buffer]).to(self.device)
        # serialized_parameters = Aggregators.fedavg_aggregate(parameters_list, weights)
        # SerializationTool.deserialize_model(self._model, serialized_parameters)
    def set_momentum(self):
        SerializationTool.deserialize_model(self.delta, self.delta_parameters)

    def calc_momentum(self, buffer):
        # parameters_list = [ele[0] for ele in buffer]
        # weights = torch.tensor(weights)
        # weights = weights / torch.sum(weights)
        # S = len(buffer)

        eta_l = self.eta_l
        weights = [ele[1] for ele in buffer]
        K = np.array(weights).mean()

        # K = self.K
        # K = buffer[0][2]    # number of epoch
        gradient_list = [
            torch.sub(ele[0], self.model_parameters) for idx, ele in enumerate(buffer)
        ]

        delta = torch.mean(torch.stack(gradient_list, dim=0), dim=0)
        delta.div_(-1*eta_l*K)
        return delta

##################
#
#      Client
#
##################


class MoFedSamSerialClientTrainer(SGDSerialClientTrainer):
    def __init__(self, model, num_clients, rho, beta, cuda=True, device=None, logger=None, personal=False) -> None:
        super().__init__(model, num_clients, cuda, device, logger, personal)
        self.rho = rho
        self.beta = beta

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        delta = payload[1]
        # model_parameters_np = payload[2]

        for id in id_list:
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            minimizer = MoSAM(self.optimizer, self.model, self.rho, self.beta, delta)
            pack = self.train(id, model_parameters, minimizer, data_loader)
            self.cache.append(pack)

    def train(self, id, model_parameters, minimizer, train_loader):
        self.set_model(model_parameters)

        # data_size = 0
        num_update = 0
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

                # data_size += len(target)
                num_update += 1

        return [self.model_parameters, num_update]
