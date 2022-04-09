import gym
from gym.spaces import Box
import numpy as np
from PIL import Image as im

env = gym.make('CarRacing-v1')
action_space = env.action_space
env.observation_space = Box(low=0, high=255, shape=(3,64,64), dtype=env.observation_space.dtype)

env.reset()
random_action = env.action_space.sample()
obs, reward, done, info = env.step(random_action)
transposed_arr = np.transpose(obs, (2,0,1))
im.fromarray(transposed_arr, 'RGB').show()
print('')