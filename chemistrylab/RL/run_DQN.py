from DQN_algo import DeepQNetwork
import gym
import chemistrylab


import numpy as np
from time import sleep
import time
import csv 

with open('DQN.csv', 'w+') as myfile:
    myfile.write('{0},{1}\n'.format("Episode", "Reward"))


env = gym.make('ExtractWorld_0-v1')

n_features = 1548
RL = DeepQNetwork(env.action_space, n_features,
                  learning_rate=0.01,
                  reward_decay=0.9,
                  e_greedy=0.9,
                  replace_target_iter=200,
                  memory_size=2000,
                  # output_graph=True
                  )


step = 0

def changeobservation(obs):
    new_list = []
    for i in range(len(obs)):
        for j in range(len(obs[i])):
            for k in range(len(obs[i][j])):
                tmpshape = obs[i][j][k].shape
                if j == 2: 
                    new_list.append(obs[i][j][k])
                else: 
                    for s in range(tmpshape[0]):
                        new_list.append(obs[i][j][k][s])

    new_list = np.asarray(new_list)
    return new_list



for episode in range(500):
    observation = env.reset()
    total_reward = 0
    while True:
        env.render()
        observation_newformat = changeobservation(observation)
        new_action, action = RL.choose_action(observation_newformat)

        observation_, reward, done, _ = env.step(new_action)
        total_reward = total_reward + reward
        observation_newformat_ = changeobservation(observation_)
        RL.store_transition(observation_newformat, action, reward, observation_newformat_)

        if (step > 200) and (step % 5 == 0):
            RL.learn()

        observation = observation_

        if done:
            break
        step += 1
    
    with open('DQN.csv', 'a') as myfile:
        myfile.write('{0},{1}\n'.format(episode, total_reward))
    print("Writen to file")


print('game over')


