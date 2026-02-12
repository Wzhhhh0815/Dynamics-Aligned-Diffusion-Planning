import gym
import torch
import torch.nn.functional as F
from torch.optim import Adam
from networks import VAE
import d4rl
from buffer import ReplayBuffer
import matplotlib.pyplot as plt
import os
import numpy as np

def fill_replay_buffer(dataset, replay_buffer):
    observations = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    terminals = dataset['terminals']
    next_observations = observations[1:]
    next_observations = np.append(next_observations, [np.zeros_like(observations[0])], axis=0)  # pad with zeros
    print('buffer size:', len(observations))
    for i in range(len(observations) - 1):
        replay_buffer.add_sample(
            observation=observations[i],
            action=actions[i],
            reward=rewards[i],
            next_observation=next_observations[i],
            terminal=terminals[i]
        )

def np_to_pytorch_batch(data):
    # Convert numpy arrays to PyTorch tensors
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()}

class DynamicsTrainer:
    def __init__(self, env, device, batch_size=64):
        self.env = env
        self.device = device
        self.batch_size = batch_size
        self.loss_history = []
        self.recon_loss_history = []
        self.KL_loss_history = []
        # Inverse dynamics model: VAE
        self.dynamics = VAE(env.observation_space.shape[0] + env.observation_space.shape[0],
                            env.action_space.shape[0],
                            device).to(device)
        self.dynamics_optimizer = Adam(self.dynamics.parameters(), lr=0.001)
        self.replay_buffer = replay_buffer

    def save_model(self, filename):
        model_dir = f'inverse_dynamics_model/models/{self.env.unwrapped.spec.id}'
        os.makedirs(model_dir, exist_ok=True)
        filepath = os.path.join(model_dir, filename)
        torch.save(self.dynamics.state_dict(), filepath)

    def plot_loss(self):
        plt.figure(figsize=(15, 10))
        plt.subplot(3, 1, 1)
        plt.plot(self.recon_loss_history, label='Reconstruction Loss', color='blue')
        plt.title('Reconstruction Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Reconstruction Loss')
        plt.legend()
        plt.grid(True)
        plt.subplot(3, 1, 2)
        plt.plot(self.KL_loss_history, label='KL Loss', color='red')
        plt.title('KL Divergence Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('KL Loss')
        plt.legend()
        plt.grid(True)
        plt.subplot(3, 1, 3)
        plt.plot(self.loss_history, label='VAE Total Loss', color='green')
        plt.title('VAE Total Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('VAE Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        model_dir = f'inverse_dynamics_model/models/{self.env.unwrapped.spec.id}'
        os.makedirs(model_dir, exist_ok=True)
        loss_plot_filename = os.path.join(model_dir, 'training_loss_components.png')
        plt.savefig(loss_plot_filename)
        print(f"Loss plot saved to {loss_plot_filename}")
        plt.close()

    def train_dynamics(self, batch):
        self.dynamics.train()
        observations = batch['observations'].to(self.device)
        actions = batch['actions'].to(self.device)
        next_observations = batch['next_observations'].to(self.device)
        recon, mean, std = self.dynamics(observations, next_observations)
        recon_loss = F.mse_loss(recon, actions, reduce=False).mean()
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * KL_loss
        self.dynamics_optimizer.zero_grad()
        vae_loss.backward()
        self.dynamics_optimizer.step()
        self.recon_loss_history.append(recon_loss.item())
        self.KL_loss_history.append(KL_loss.item())
        return vae_loss.item()

    def train(self, epoch, save_interval=10):
        for i in range(epoch):
            train_data, indices = self.replay_buffer.random_batch(self.batch_size, return_indices=True)
            batch = np_to_pytorch_batch(train_data)
            loss = self.train_dynamics(batch)
            self.loss_history.append(loss)
            print(f"Epoch {i}: dynamics loss: {loss}")
            if (i + 1) % save_interval == 0 or i == epoch - 1:
                self.save_model(f'epoch_{i + 1}.pth')
        self.plot_loss()

# Load environment and dataset
env = gym.make('halfcheetah-medium-expert-v2')
dataset = env.get_dataset()
observation_dim = dataset['observations'].shape[1]
action_dim = dataset['actions'].shape[1]
# Create and fill replay buffer
replay_buffer = ReplayBuffer(len(dataset['observations']), observation_dim, action_dim)
fill_replay_buffer(dataset, replay_buffer)

trainer = DynamicsTrainer(env, 'cuda:3')
trainer.train(5000, save_interval=1000)
