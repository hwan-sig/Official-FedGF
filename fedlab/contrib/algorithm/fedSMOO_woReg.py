import torch
import copy
from collections import OrderedDict

from .basic_server import SyncServerHandler
from .basic_client import SGDClientTrainer, SGDSerialClientTrainer
from .fedavg import FedAvgServerHandler
from .minimizers import SAM
from ...utils import Aggregators
from .bypass_bn import disable_running_stats, enable_running_stats
from statistics import mean
from collections import defaultdict

##################
#
#      Server
#
##################


# class FedSAMServerHandler(SyncServerHandler):
class FedSMOONoRegServerHandler(FedAvgServerHandler):

    @property
    def downlink_package(self):
        return [self.model_parameters, self.s]

    def setup_optim(self, rho):
        self.rho = rho
        self.s = torch.zeros_like(self.model_parameters)

    def global_update(self, buffer, upload_res=False):

        self.s = self.calc_s(buffer)
        super().global_update(buffer)

    def calc_s(self, buffer):
        parameters_list = [ele[2] for ele in buffer]
        weights = torch.ones(len(parameters_list)).cuda()
        weights = weights / torch.sum(weights)

        serialized_parameters = torch.sum(torch.stack(parameters_list, dim=-1) / weights, dim=-1)
        return self.rho * serialized_parameters / serialized_parameters.norm()

##################
#
#      Client
#
##################


class FedSMOONoRegSerialClientTrainer(SGDSerialClientTrainer):
    def __init__(self, model, num_clients, rho, cuda=True, device=None, logger=None, personal=False) -> None:
        super().__init__(model, num_clients, cuda, device, logger, personal)
        self.mu_i = [torch.zeros_like(self.model_parameters) for _ in range(num_clients)]
        self.rho = rho

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        s = payload[1]

        for id in id_list:
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            pack = self.train(id, model_parameters, data_loader, s)
            self.cache.append(pack)

    def train(self, id, model_parameters, train_loader, s):
        self.set_model(model_parameters)
        hat_s = None
        mu_i = copy.deepcopy(self.mu_i[id])

        data_size = 0
        for _ in range(self.epochs):
            for data, target in train_loader:
                origin_param = self.model_parameters
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                data_size += len(target)

                self.optimizer.zero_grad()
                loss.backward()

                tier = self.model_gradients - mu_i - s
                hat_s = self.calc_hats(tier)
                mu_i = mu_i + hat_s - s

                self.optimizer.zero_grad()

                self.set_model(self.model_parameters+hat_s)
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()

                self.set_model(origin_param)

                self.optimizer.step()

                data_size += len(target)
        tilde_si = mu_i - hat_s
        self.mu_i[id] = mu_i

        return [self.model_parameters, data_size, tilde_si]

    def calc_hats(self, tier):
        return self.rho * tier / tier.norm()
