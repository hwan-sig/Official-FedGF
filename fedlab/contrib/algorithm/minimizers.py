import torch
from collections import defaultdict

class ASAM:
    def __init__(self, optimizer, model, rho=0.5, eta=0.01):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)

    @torch.no_grad()
    def ascent_step(self):
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if 'weight' in n:
                t_w[...] = p[...]
                t_w.abs_().add_(self.eta)
                p.grad.mul_(t_w)
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if 'weight' in n:
                p.grad.mul_(t_w)
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(self.rho / wgrad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self, init_model=None):
        if init_model is None:
            for n, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()

class SAM(ASAM):
    @torch.no_grad()
    def ascent_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()


class MoSAM(SAM):
    def __init__(self, optimizer, model, rho, beta, delta):
        super().__init__(optimizer, model, rho)
        self.beta = beta
        self.delta = delta
        # self.model_parameters_np = model_parameters_np

    @torch.no_grad()
    def descent_step(self):
        idx = 0
        for n, p in self.model.named_parameters():
            layer_size = p.grad.numel()
            shape = p.grad.shape

            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])

            p.grad.mul_(self.beta)
            momentum_grad = self.delta[idx:idx + layer_size].view(shape)[:]
            momentum_grad = momentum_grad.mul_(1 - self.beta).cuda()

            p.grad.add_(momentum_grad)

            idx += layer_size
        self.optimizer.step()
        self.optimizer.zero_grad()
class GF_ADMM(SAM):
    @torch.no_grad()
    def descent_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()