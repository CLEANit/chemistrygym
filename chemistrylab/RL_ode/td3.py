import gym
import numpy as np
import csv
from stable_baselines import TD3
from stable_baselines.td3.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.noise import NormalActionNoise
import chemistrylab

env = gym.make('ReactionBench_0-v0')
render_mode = 'human'
action_noise = NormalActionNoise(np.zeros(1), 0.1 * np.ones(1))
with open('td3.csv', 'w+') as myfile:
    myfile.write('{0},{1}\n'.format("Episode", "Reward"))


n_actions = env.action_space.shape[-1]

model = TD3(MlpPolicy, env, action_noise = action_noise, verbose=1)
model.learn(total_timesteps=50000, log_interval=10)
model.save("td3_odeworld")

del model # remove to demonstrate saving and loading

model = TD3.load("td3_odeworld")

obs = env.reset()
episode = 0
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    with open('td3.csv', 'a') as myfile:
        myfile.write('{0},{1}\n'.format(episode, rewards))
    print("Writen to file")
    episode = episode + 1
    env.render()
