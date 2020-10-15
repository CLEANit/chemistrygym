import chemistrylab
from time import sleep
import time
import csv

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import gym

np.random.seed(2)
tf.set_random_seed(2)  


with open('A2C.csv', 'w+') as myfile:
    myfile.write('{0},{1}\n'.format("Episode", "Reward"))

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 100
DISPLAY_REWARD_THRESHOLD = 200 
MAX_EP_STEPS = 1000   
RENDER = False  
GAMMA = 0.9     
LR_A = 0.001    
LR_C = 0.01     

N_F = 1548
env = gym.make('ExtractWorld_0-v1')
n_actions = env.action_space
action_dict = {}
k = 0
for i in range(n_actions.nvec[0]):
    for j in range(n_actions.nvec[1]):
        temp_list = []
        temp_list.append(i)
        temp_list.append(j)
        action_dict[k] = temp_list
        k = k + 1

N_A = len(action_dict)


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




class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        action = np.random.choice(np.arange(probs.shape[1]), p=probs.ravel()) 
        new_action = action_dict[action]
        new_action = np.asarray(new_action)
        return new_action, action


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units
                activation=tf.nn.relu,  # None
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error


sess = tf.Session()

actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(sess, n_features=N_F, lr=LR_C)    
sess.run(tf.global_variables_initializer())

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)

for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER: env.render()
        
        s_newformat = changeobservation(s)
        new_a, a = actor.choose_action(s_newformat)

        s_, r, done, info = env.step(new_a)


        track_r.append(r)
        s_newformat_ = changeobservation(s_)
        td_error = critic.learn(s_newformat, r, s_newformat_)  
        actor.learn(s_newformat, a, td_error)     
        s = s_
        t += 1


        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            running_reward = ep_rs_sum
            print("episode:", i_episode, "  reward:", int(running_reward))
            with open('A2C.csv', 'a') as myfile:
                myfile.write('{0},{1}\n'.format(i_episode, running_reward))
                print("Writen to file")
            break

