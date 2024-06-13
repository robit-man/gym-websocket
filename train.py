import gymnasium as gym
from stable_baselines3 import PPO
import custom_bipedal_walker  # Ensure this import registers the custom environment
import torch

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())

# Create the custom environment
env = gym.make('CustomBipedalWalker-v0')

# Initialize the PPO model with GPU support
model = PPO('MlpPolicy', env, verbose=1, device='cuda')

# Print model device
print("Model device:", model.device)

# Train the model
model.learn(total_timesteps=1000000)

# Save the model
model.save("ppo_custom_bipedal_walker")

# Close the environment
env.close()
