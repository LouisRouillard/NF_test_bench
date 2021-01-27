# Refs:
# https://github.com/clbonet/Generative-Models/tree/main/Normalizing%20Flows
# https://github.com/acids-ircam/pytorch_flows

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as td

from abc import ABC, abstractmethod
from tqdm.auto import trange


class BaseNormalizingFlow(ABC, nn.Module):
    """
    Abtract class for NF
    """
    def __init__(self, device):
        super().__init__()
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, z):
        pass


class Shifting(nn.Module):
    def __init__(self, input_dim, nh=None, n_layers=1, device=None):
        super().__init__()
        if nh is None:
            self.nh = input_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, self.nh))
        for i in range(n_layers):
            self.layers.append(nn.Linear(self.nh, self.nh))
        self.layers.append(nn.Linear(self.nh, input_dim))
        self.layers.to(device)

    def forward(self, x):
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.2)
        return x


class Scaling(nn.Module):
    def __init__(self, input_dim, nh=None, n_layers=1, device=None):
        super().__init__()
        if nh is None:
            self.nh = input_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, self.nh))
        for i in range(n_layers):
            self.layers.append(nn.Linear(self.nh, self.nh))
        self.layers.append(nn.Linear(self.nh, input_dim))
        self.layers.to(device)

    def forward(self, x):
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x


class AffineCoupling(BaseNormalizingFlow):
    def __init__(self, input_dim, device=None):
        super().__init__(device)
        self.scaling = Scaling(input_dim // 2, device=self.device)
        self.shifting = Shifting(input_dim // 2, device=self.device)
        self.k = input_dim // 2

    def forward(self, x):
        x0, x1 = x[:, :self.k], x[:, self.k:]

        s = self.scaling(x0)
        t = self.shifting(x0)
        z0 = x0
        z1 = torch.exp(s)*x1+t

        z = torch.cat([z0, z1], dim=1)
        return z, torch.sum(s, dim=1)

    def backward(self, z):
        z0, z1 = z[:, :self.k], z[:, self.k:]

        s = self.scaling(z0)
        t = self.shifting(z0)
        x0 = z0
        x1 = torch.exp(-s)*(z1-t)

        x = torch.cat([x0, x1], dim=1)
        return x, -torch.sum(s, dim=1)


class Reverse(BaseNormalizingFlow):
    def __init__(self, input_dim, device=None):
        super().__init__(device)
        self.permute = torch.arange(input_dim - 1, -1, -1)
        self.inverse = torch.argsort(self.permute)

    def forward(self, x):
        return x[:, self.permute], torch.zeros(x.size(0), device=self.device)

    def backward(self, z):
        return z[:, self.inverse], torch.zeros(z.size(0), device=self.device)


class BatchNorm(BaseNormalizingFlow):
    def __init__(self, d, eps=1e-5, momentum=0.95, device=None):
        super().__init__(device)
        self.eps = eps
        self.momentum = momentum
        self.train_mean = torch.zeros(d, device=self.device)
        self.train_var = torch.ones(d, device=self.device)

        self.gamma = nn.Parameter(torch.ones(d, device=self.device))
        self.beta = nn.Parameter(torch.ones(d, device=self.device))

    def forward(self, x):
        if self.training:
            self.batch_mean = x.mean(0)
            self.batch_var = (x - self.batch_mean).pow(2).mean(0) + self.eps

            self.train_mean = self.momentum * self.train_mean + (1 - self.momentum) * self.batch_mean
            self.train_var = self.momentum * self.train_var + (1 - self.momentum) * self.batch_var

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.train_mean
            var = self.train_var

        z = torch.exp(self.gamma) * (x - mean) / var.sqrt() + self.beta
        log_det = torch.sum(self.gamma - 0.5 * torch.log(var))
        return z, log_det

    def backward(self, z):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.train_mean
            var = self.train_var

        x = (z - self.beta) * torch.exp(-self.gamma) * var.sqrt() + mean
        log_det = torch.sum(-self.gamma + torch.log(var))
        return x, log_det


class NormalizingFlows(BaseNormalizingFlow):
    def __init__(self, flows, device=None):
        super().__init__(device)
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        log_det = torch.zeros(x.shape[0], device=self.device)
        zs = [x]
        for flow in self.flows:
            x, log_det_i = flow(x)
            log_det += log_det_i
            zs.append(x)
        return zs, log_det

    def backward(self, z):
        log_det = torch.zeros(z.shape[0], device=self.device)
        xs = [z]
        for flow in self.flows[::-1]:
            z, log_det_i = flow.backward(z)
            log_det += log_det_i
            xs.append(z)
        return xs, log_det


class RealNVP(BaseNormalizingFlow):
    """
    Real NVP
    """
    def __init__(self, input_dim, depth=5, device=None):
        super().__init__(device)
        self.input_dim = input_dim
        flows = []
        for i in range(depth):
            flows.append(AffineCoupling(input_dim, device=self.device))
            flows.append(Reverse(input_dim, device=self.device))
            flows.append(BatchNorm(input_dim, device=self.device))
        self.model = NormalizingFlows(flows)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=1e-4,
                                    weight_decay=1e-5)
        self.prior = td.multivariate_normal.MultivariateNormal(
            torch.zeros(input_dim, device=self.device),
            torch.eye(input_dim, device=self.device))
        print("number of params: ", sum(p.numel() for p in self.model.parameters()))

    def forward(self, x):
        return self.model.forward(x)

    def backward(self, z):
        return self.model.backward(z)

    def loss(self, z, log_det):
        log_prior = self.prior.log_prob(z)
        return -(log_prior + log_det).mean()

    def train(self, trainloader, epochs=50):
        print_interval = 100
        pbar = trange(epochs)
        for epoch in pbar:
            for i, (x_batch, _) in enumerate(trainloader):
                x_batch = x_batch.to(self.device)
                z, log_det = self.forward(x_batch)
                loss = self.loss(z[-1], log_det)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if i % print_interval == 0:
                    pbar.set_postfix_str(f"loss = {loss.item():.3f}")
            if epoch % 5 == 0:
                self.plot_generations()

    def plot_generations(self, num_samples=10):
        lambd = 1e-6
        z_sample = torch.randn(num_samples, self.input_dim).to(self.device)
        x_gen, _ = self.backward(z_sample)
        x_gen = (torch.sigmoid(x_gen[-1]) - lambd) / (1 - 2 * lambd)
        x_gen = x_gen.cpu().detach().numpy().reshape(-1, 28, 28)

        fig, axes = plt.subplots(nrows=1,
                                 ncols=num_samples,
                                 figsize=(num_samples, 1))

        for i in range(num_samples):
            axes[i].imshow(x_gen[i], cmap="gray")
            axes[i].axis("off")

        plt.show()
