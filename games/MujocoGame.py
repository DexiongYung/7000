import gym
import numpy as np
from utils.obs_processor import process_obs

class MujocoGame:
    def __init__(self, cfg: dict, seed: int) -> None:
        self.env = gym.make(cfg['task'])
        self.env.seed = seed
        self.seed = seed
        self.rewards = list()
        self.obs = list()
        self.width=cfg['screen_width']
        self.height=cfg['screen_height']
    
    def step(self, action: np.array):
        _, reward, done, info = self.env.step(action=action)
        rgb_obs = self.env.render(mode='rgb_array', width=self.width, height=self.height)
        rgb_obs_tnsr = process_obs(obs=rgb_obs)
        
        self.rewards.append(reward)
        self.obs.append(rgb_obs)

        return rgb_obs_tnsr, reward, done, info
    
    def reset(self):
        self.env.reset()

        assert self.env.seed == self.seed
        
        rgb_obs = self.env.render(mode='rgb_array', width=self.width, height=self.height)
        rgb_obs_tnsr = process_obs(obs=rgb_obs)
        self.obs = list()
        self.rewards = list()

        return rgb_obs_tnsr