from qltabular_algo import QLearningTable
import gym
import chemistrylab
import numpy as np
from time import sleep
import time
import csv

#env_dict = gym.envs.registration.registry.env_specs.copy()
#for env in env_dict:
#    if 'ExtractWorld_0-v1' in env:
#        print("Remove {} from registry".format(env))
#        del gym.envs.registration.registry.env_specs[env]
env = gym.make('ExtractWorld_0-v1')
RL = QLearningTable(actions=env.action_space)
with open('tabularQ.csv', 'w+') as myfile:
    myfile.write('{0},{1}\n'.format("Episode", "Reward"))
for episode in range(500):
    observation = env.reset()
    total_reward = 0
    while True:
        env.render()
        new_action, action = RL.choose_action(str(observation))

        observation_, reward, done, _ = env.step(new_action)
        
        total_reward = total_reward + reward 
        RL.learn(str(observation), action, reward, str(observation_))

        observation = observation_
        
        
        if done:
            break

    with open('tabularQ.csv', 'a') as myfile:
        myfile.write('{0},{1}\n'.format(episode, total_reward))
    print("Writen to file")


print('game over')


