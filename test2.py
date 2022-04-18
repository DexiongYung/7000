import gym
from PIL import Image as im
from models.PPO_model import PPO_model
from utils.tensor_utils import process_obs

WIDTH = 64
HEIGHT = WIDTH

env = gym.make('Hopper-v3')
action_space = env.action_space
env.reset()
rgb_obs = env.render(mode='rgb_array', width=WIDTH, height=HEIGHT)
model = PPO_model(input_width=WIDTH, input_height=HEIGHT, action_space=action_space, is_discrete=False)

for i in range(1):
    random_action = env.action_space.sample()
    _, reward, done, info = env.step(random_action)
    rgb_obs = env.render(mode='rgb_array', width=WIDTH, height=HEIGHT)
    transformed_obs = process_obs(obs=rgb_obs, model=model)
    pi, v = model.forward(transformed_obs)
    print('')