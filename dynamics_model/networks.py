import numpy as np
import torch
from torch import nn as nn
from torch.distributions import Normal, TransformedDistribution, TanhTransform
from torch.nn import functional as F

def identity(x):
    return x

class VAE(nn.Module):
    def __init__(self, input_dim, output_dim, device,
                 log_std_multiplier=1.0, log_std_offset=-1.0, no_tanh=False):
        super(VAE, self).__init__()
        self.latent_dim = 32
        dim =  256
        self.device = device

        self.e1 = nn.Linear(input_dim, dim)
        self.e2 = nn.Linear(dim, dim)
        self.e3 = nn.Linear(dim, dim)

        self.mean = nn.Linear(dim, self.latent_dim)
        self.log_std = nn.Linear(dim, self.latent_dim)

        self.d1 = nn.Linear(input_dim + self.latent_dim, dim)
        self.d2 = nn.Linear(dim, dim)
        self.d3 = nn.Linear(dim, output_dim)

        self.output_dim = output_dim

    def forward(self, observation, next_observation):

        self.z1 = torch.relu(self.e1(torch.cat([observation, next_observation], dim=-1)))
        self.z2 = torch.relu(self.e2(self.z1))
        self.z3 = torch.relu(self.e3(self.z2))

        mean = self.mean(self.z3)
        log_std = self.log_std(self.z3)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std).to(self.device)
        u = self.decode2(observation, next_observation, z)
        return u, mean, std

    def decode2(self, observation, next_observation, z=None, deterministic=False):
        if z is None:
            dis = torch.distributions.Normal(torch.zeros(self.latent_dim),torch.ones(self.latent_dim))
            z = dis.sample((observation.shape[0],)).clamp(-0.5, 0.5).to(self.device)
        a1 = torch.relu(self.d1(torch.cat([observation, next_observation, z], 1)))
        a2 = torch.relu(self.d2(a1))
        output = self.d3(a2)
        return output