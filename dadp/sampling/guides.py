import torch
import torch.nn as nn
import pdb


class ValueGuide(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, cond, t):
        output = self.model(x, cond, t)
        return output.squeeze(dim=-1)

    def gradients(self, x, *args):
        x.requires_grad_()
        y = self(x, *args)
        grad = torch.autograd.grad([y.sum()], [x])[0]
        x.detach()
        return y, grad

class ValueGuide_Inverse(nn.Module):

    def __init__(self, model, dynamics_model, env):
        super().__init__()
        self.model = model
        self.dynamics_model = dynamics_model
        self.env = env

    def forward(self, x, cond, t):
        output = self.model(x, cond, t)
        return output.squeeze(dim=-1)

    def gradients_values(self, x, *args):
        x.requires_grad_()
        y = self(x, *args)
        grad = torch.autograd.grad([y.sum()], [x])[0]
        x.detach()
        return y, grad

    def gradients_deviation(self, x, *args):
        observations = x[:, :-1, self.env.action_space.shape[0]:]
        actions = x[:, :-1, :self.env.action_space.shape[0]]
        targets = x[:, 1:, self.env.action_space.shape[0]:]
        batch_size, horizon, _ = observations.shape
        observations = observations.reshape(batch_size * horizon, -1)
        actions = actions.reshape(batch_size * horizon, -1)
        targets = targets.reshape(batch_size * horizon, -1)

        predictions = self.dynamics_model(observations,targets)[0]
        mse_loss = nn.MSELoss(reduction='none')(actions, predictions.detach()).mean(dim=1)
        grad = torch.autograd.grad([mse_loss.sum()], [x])[0]
        # x.detach()

        mse_loss = mse_loss.view(batch_size, horizon).mean(dim=0)
        return grad, mse_loss
