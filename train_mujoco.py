import gym
import logging
from utils import setup_logger_and_config, device

def train(cfg):
    env = gym.make(cfg['task'])
    action_space = env.action_space
    
    env.reset()
    rgb_obs = env.render(mode='rgb_array', width=cfg['screen_width'], height=cfg['screen_height'])

    for i in range(cfg['train']['steps']):
        action = None # Forward rgb_obs through model to get action model.forward(rgb_obs)
        _, reward, done, _ = env.step(action)

        if done:
            env.reset()
        
        rgb_obs = env.render(mode='rgb_array', width=cfg['screen_width'], height=cfg['screen_height'])

if __name__ == '__main__':
    cfg = setup_logger_and_config()
    train(cfg)