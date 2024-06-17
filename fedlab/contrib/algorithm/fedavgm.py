import torch

# from utils_algorithms import MinNormSolver
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.contrib.algorithm.basic_server import SyncServerHandler


class FedAvgMServerHandler(SyncServerHandler):
    """
    Hsu, Tzu-Ming Harry, Hang Qi, and Matthew Brown. "Measuring the effects of non-identical data distribution for federated visual classification." arXiv preprint arXiv:1909.06335 (2019).
    """

    def setup_optim(self, beta):
        self.momentum = torch.zeros_like(self.model_parameters)
        self.beta = beta
    # def setup_optim(self, epochs, batch_size, lr, weight_decay, momentum, beta):
    #     super().setup_optim(epochs, batch_size, lr, weight_decay, momentum)
    #     # self.n = self.num_clients
        # self.num_to_sample = int(self.sample_ratio * self.n)
        # self.round_clients = int(self.sample_ratio * self.n)
        # self.sampler = sampler

        # self.args = args
        # self.lr = args.glr
        #self.k = args.k


    # @property
    # def num_clients_per_round(self):
    #     return self.round_clients

    # def sample_clients(self, num_to_sample=None):
    #     clients = self.sampler.sample(self.num_to_sample)
    #     self.round_clients = len(clients)
    #     assert self.num_clients_per_round == len(clients)
    #     return clients

    def global_update(self, buffer):
        gradient_list = [
            torch.sub(self.model_parameters, ele[0]) for ele in buffer
        ]
        weights = [ele[1] for ele in buffer]

        # indices, _ = self.sampler.last_sampled
        estimates = Aggregators.fedavg_aggregate(gradient_list,
                                                 weights)
        self.momentum = self.beta * self.momentum + estimates

        serialized_parameters = self.model_parameters - self.momentum
        SerializationTool.deserialize_model(self._model, serialized_parameters)
