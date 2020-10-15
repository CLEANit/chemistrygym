import gym
import numpy as np

from stable_baselines import TD3
from stable_baselines.td3.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
import chemistrylab

env = gym.make('ReactionBench_0-v0')
render_mode = 'human'

# The noise objects for TD3
n_actions = env.action_space.shape[-1]

model = TD3(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=50000, log_interval=10)
model.save("td3_odeworld")

del model # remove to demonstrate saving and loading

model = TD3.load("td3_odeworld")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
