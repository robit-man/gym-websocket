import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.bipedal_walker import BipedalWalker

class CustomBipedalWalker(BipedalWalker):
    def __init__(self):
        super(CustomBipedalWalker, self).__init__()

    def step(self, action):
        obs, reward, done, truncated, info = super(CustomBipedalWalker, self).step(action)

        # Joint limits in radians (30 degrees for hips, 60 degrees for knees and ankles)
        joint_limits = {
            'hip_yaw': 0.524,  # 30 degrees
            'hip_roll': 0.524, # 30 degrees
            'hip_pitch': 0.524, # 30 degrees
            'knee_pitch': 1.047, # 60 degrees
            'ankle_pitch': 1.047 # 60 degrees
        }

        penalties = 0

        # Penalize hip yaw motion heavily
        penalties += self.calculate_penalty(obs[0], joint_limits['hip_yaw'], heavy=True)
        penalties += self.calculate_penalty(obs[3], joint_limits['hip_yaw'], heavy=True)

        # Penalize other joint motions
        penalties += self.calculate_penalty(obs[1], joint_limits['hip_roll'])
        penalties += self.calculate_penalty(obs[2], joint_limits['hip_pitch'])
        penalties += self.calculate_penalty(obs[4], joint_limits['hip_roll'])
        penalties += self.calculate_penalty(obs[5], joint_limits['hip_pitch'])
        penalties += self.calculate_penalty(obs[6], joint_limits['knee_pitch'])
        penalties += self.calculate_penalty(obs[7], joint_limits['knee_pitch'])
        penalties += self.calculate_penalty(obs[8], joint_limits['ankle_pitch'])
        penalties += self.calculate_penalty(obs[9], joint_limits['ankle_pitch'])

        # Encourage smooth transitions and realistic walking
        movement_penalty = np.sum(np.abs(np.diff(action)))
        smoothness_reward = -0.1 * movement_penalty
        realistic_motion_reward = 1.0 - penalties

        reward += smoothness_reward + realistic_motion_reward

        return obs, reward, done, truncated, info

    def calculate_penalty(self, angle, limit, heavy=False):
        if abs(angle) > limit:
            if heavy:
                return 2 * (abs(angle) - limit)  # Heavier penalty for yaw
            else:
                return abs(angle) - limit
        return 0

gym.envs.register(
    id='CustomBipedalWalker-v0',
    entry_point='custom_bipedal_walker:CustomBipedalWalker',
    max_episode_steps=2000,
)
