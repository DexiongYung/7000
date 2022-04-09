from dm_control import suite
import numpy as np

random_state = np.random.RandomState(42)
env = suite.load('hopper', 'stand', task_kwargs={'random': random_state})

# Simulate episode with random actions
duration = 4  # Seconds
frames = []
ticks = []
rewards = []
observations = []

spec = env.action_spec()
time_step = env.reset()

while env.physics.data.time < duration:

  action = random_state.uniform(spec.minimum, spec.maximum, spec.shape)
  time_step = env.step(action)
  obs = time_step.observation
  camera0 = env.physics.render(camera_id=0, height=200, width=200)
  pixels = env.physics.render(height=200, width=200, camera_id=0)
  print('')