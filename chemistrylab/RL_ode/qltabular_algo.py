import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.action_dict = {}
        k = 0
        for i in range(actions.nvec[0]):
            for j in range(actions.nvec[1]):
                temp_list = []
                temp_list.append(i)
                temp_list.append(j)
                self.action_dict[k] = temp_list
                k = k + 1
        
        action_list = []
        for i in range(len(self.action_dict)):
            action_list.append(i)
        
        self.actions = action_list
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        new_action = self.action_dict[action]
        new_action = np.asarray(new_action)
        
        return new_action, action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
