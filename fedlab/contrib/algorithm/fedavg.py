from .basic_server import SyncServerHandler
from .basic_client import SGDClientTrainer, SGDSerialClientTrainer
from ...utils.aggregator import Aggregators
from ...utils.serialization import SerializationTool
import torch
import copy

##################
#
#      Server
#
##################


class FedAvgServerHandler(SyncServerHandler):
    """FedAvg server handler."""
    def global_update(self, buffer, upload_res=False):
        parameters_list = [ele[0] for ele in buffer]
        weights = torch.tensor([ele[1] for ele in buffer]).to(self.device)
        serialized_parameters = Aggregators.fedavg_aggregate(parameters_list, weights)
        SerializationTool.deserialize_model(self._model, serialized_parameters)

    def setup_swa_model(self):
        self.swa_model = copy.deepcopy(self.model_parameters)

    def update_swa_model(self, alpha):
        self.swa_model *= (1.0 - alpha)
        self.swa_model += self.model_parameters * alpha
        # for param1, param2 in zip(self.swa_model, self.model_parameters):
        #     param1.data *= (1.0 - alpha)
        #     param1.data += param2.data * alpha

    def update_clients_lr(self, lr, clients=None):
        if clients is None:
            clients = self.round_clients
        for c in clients:
            c.update_lr(lr)


##################
#
#      Client
#
##################


class FedAvgClientTrainer(SGDClientTrainer):
    """Federated client with local SGD solver."""
    def global_update(self, buffer):
        parameters_list = [ele[0] for ele in buffer]
        weights = [ele[1] for ele in buffer]
        serialized_parameters = Aggregators.fedavg_aggregate(
            parameters_list, weights)
        SerializationTool.deserialize_model(self._model, serialized_parameters)


class FedAvgSerialClientTrainer(SGDSerialClientTrainer):
    """Federated client with local SGD solver."""
    def train(self, model_parameters, train_loader):
        self.set_model(model_parameters)
        self._model.train()

        data_size = 0
        for _ in range(self.epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                data_size += len(target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return [self.model_parameters, data_size]
