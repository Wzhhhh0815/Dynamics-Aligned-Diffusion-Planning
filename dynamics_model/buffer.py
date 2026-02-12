import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, observation_dim, action_dim):
        self.max_size = max_size
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.ptr = 0
        self.size = 0

        self.observations = np.zeros((max_size, observation_dim))
        self.actions = np.zeros((max_size, action_dim))
        self.rewards = np.zeros(max_size)
        self.terminals = np.zeros(max_size, dtype=np.float32)
        self.next_observations = np.zeros((max_size, observation_dim))

    def add_sample(self, observation, action, reward, next_observation, terminal):
        idx = self.ptr % self.max_size
        self.observations[idx] = observation
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_observations[idx] = next_observation
        self.terminals[idx] = terminal

        self.ptr += 1
        self.size = min(self.size + 1, self.max_size)

    def random_batch(self, batch_size, min_pct=0, max_pct=1, include_logprobs=False, return_indices=False):
        indices = np.random.randint(int(min_pct * self.size), int(max_pct * self.size), size=batch_size)
        batch = {
            'observations': self.observations[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'terminals': self.terminals[indices],
            'next_observations': self.next_observations[indices]
        }
        if include_logprobs and hasattr(self, 'logprobs'):
            batch['logprobs'] = self.logprobs[indices]
        if return_indices:
            return batch, indices
        return batch

    def __len__(self):
        return self.size
