from models.dynamics_network import VAE
import gym
import torch

# Load halfcheetah-expert-v2 environment
env = gym.make('halfcheetah-expert-v2')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Create model instance
model = VAE(env.observation_space.shape[0] + env.action_space.shape[0],
                            env.observation_space.shape[0], device).to(device)

# Load model parameters
model.load_state_dict(torch.load('dynamics_model/models/halfcheetah-expert-v2/epoch_5000.pth', map_location=device))

# Set model to evaluation mode
model.eval()
observation = torch.randn(5, env.observation_space.shape[0])  # Example input
action = torch.randn(5, env.action_space.shape[0])  # Example input
with torch.no_grad():  # Ensure no gradient tracking for inference
    output = model(observation.to(device),action.to(device))

print(output[0])