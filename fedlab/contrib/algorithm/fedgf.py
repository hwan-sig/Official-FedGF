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


class FedGfServerHandler(FedAvgServerHandler):
    """FedAvg server handler."""
    @property
    def downlink_package(self):
        return [self.model_parameters, self.perturbed_model_parameters, self.c]

    def setup_optim(self, g_rho, T_D, W):
        self.perturbed_model_parameters = None
        self.c = 0
        self.g_rho = g_rho

        self.window = []
        self.T_D = T_D
        self.W = W
        self.pseudo_gradient = None

    def global_update(self, buffer, upload_res=False):
        self.calc_c(buffer)
        pseudo_gradient = self.model_parameters
        super().global_update(buffer)
        # Updated average model
        pseudo_gradient.sub_(self.model_parameters)
        self.pseudo_gradient = pseudo_gradient
        self.calc_perturbation(pseudo_gradient)

    def calc_c(self, buffer):
        Divergence_metric = torch.tensor([torch.norm(torch.sub(self.model_parameters, ele[0])).item() for ele in buffer])
        tot_norm = torch.div(torch.sum(Divergence_metric), len(Divergence_metric)).item()
        self.append_grad_norm(tot_norm)
        self.norm_grad = tot_norm
        self.c = mean(self.window)

    # check return value (int, float)
    def calc_avg_norm_grad(self, parameters_list):
        total_norm = 0
        for param in parameters_list:
            total_norm += param.norm(2)
        return total_norm.item() / len(parameters_list)

    def append_grad_norm(self, grad_norm):
        x = 1 if grad_norm > self.T_D else 0
        self.window.append(x)
        if len(self.window) > self.W:
            del (self.window[0])

    def calc_perturbation(self, grad):
        # Calculate the perturbation using parameters (always)
        self.perturbed_model_parameters = copy.deepcopy(self.model_parameters)
        grad.div_(grad.norm(2)).mul_(self.g_rho)
        self.perturbed_model_parameters.add_(grad)


##################
#
#      Client
#
##################


class FedGfSerialClientTrainer(SGDSerialClientTrainer):
    def __init__(self, model, num_clients, rho, cuda=True, device=None, logger=None, personal=False) -> None:
        super().__init__(model, num_clients, cuda, device, logger, personal)
        self.rho = rho

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        perturb_parameters = payload[1]
        c = payload[2]

        for id in id_list:
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            minimizer = SAM(self.optimizer, self.model, self.rho)
            pack = self.train(id, model_parameters, minimizer, data_loader, perturb_parameters, c)
            self.cache.append(pack)

    def train(self, id, model_parameters, minimizer, train_loader, perturb_parameters, c):
        self.set_model(model_parameters)

        data_size = 0
        for _ in range(self.epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                # Ascent Step
                init_model = None

                output = self.model(data)
                if perturb_parameters is not None:
                    init_model = self.model_parameters
                loss = self.criterion(output, target)

                loss.backward()
                minimizer.ascent_step()
                if perturb_parameters is not None:
                    self.weighted_sum(perturb_parameters, c)

                # Descent Step
                self.criterion(self.model(data), target).backward()
                if perturb_parameters is not None:
                    self.set_model(init_model)

                minimizer.descent_step(init_model)

                data_size += len(target)
        return [self.model_parameters, data_size]

    def weighted_sum(self, perturb_model, c):
        """
        perturb_model에 c를 곱해  + multiply c된 global model을 1-c만큼 multiply한 local model에 더해 shift함.
        """

        model_parameters = perturb_model*c + self.model_parameters * (1-c)
        self.set_model(model_parameters)
