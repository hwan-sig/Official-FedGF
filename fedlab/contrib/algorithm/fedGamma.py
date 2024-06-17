import torch

from .basic_server import SyncServerHandler
from .basic_client import SGDClientTrainer, SGDSerialClientTrainer
from .fedavg import FedAvgServerHandler
from .minimizers import SAM
from ...utils import Aggregators

##################
#
#      Server
#
##################


class FedGammaServerHandler(FedAvgServerHandler):
    pass
    # super().__init__()
    @property
    def downlink_package(self):
        return [self.model_parameters, self.c]

    def setup_optim(self):
        self.c = torch.zeros_like(self.model_parameters)

    def global_update(self, buffer):
        weights = [ele[0] for ele in buffer]
        delta_c = [ele[2] for ele in buffer]

        avg_model = Aggregators.fedavg_aggregate(weights)
        dc = Aggregators.fedavg_aggregate(delta_c) / self.num_clients
        self.c += dc
        self.set_model(avg_model)


##################
#
#      Client
#
##################


class FedGammaSerialClientTrainer(SGDSerialClientTrainer):
    def __init__(self, model, num_clients, rho, cuda=True, device=None, logger=None, personal=False) -> None:
        super().__init__(model, num_clients, cuda, device, logger, personal)
        self.rho = rho
        self.c_i = [torch.zeros_like(self.model_parameters) for _ in range(num_clients)]

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        c = payload[1]
        for id in id_list:
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            # optimizer = torch.optim.SGD(model_parameters, lr=self.lr)
            minimizer = SAM(self.optimizer, self.model, self.rho)
            pack = self.train(id, model_parameters, minimizer, data_loader, c)
            self.cache.append(pack)

    def train(self, id, model_parameters, minimizer, train_loader, c):
        self.set_model(model_parameters)
        c_i = self.c_i[id]

        data_size = 0
        K = 0
        for _ in range(self.epochs):
            for data, target in train_loader:
                origin_param = self.model_parameters
                K += 1

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
                g_hat = self.model_gradients
                self.optimizer.zero_grad()
                grad = g_hat - c_i + c

                self.set_model(origin_param)

                current_index = 0
                for n,p in self.model.named_parameters():
                    numel = p.data.numel()
                    size = p.data.size()
                    p.grad.copy_(
                        grad[current_index:current_index + numel].view(size))
                    current_index += numel

                self.optimizer.step()
                data_size += len(target)

        delta_c_i = (model_parameters - self.model_parameters) / (self.lr * K) - c
        c_i += delta_c_i
        self.c_i[id] = c_i
        return [self.model_parameters, data_size, delta_c_i]
