import random
import numpy as np
import os
import pickle
from tqdm import trange
from queue import PriorityQueue
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    def reset(self):
        self.__init__(capacity =self.capacity)
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    def push_transition(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return {"state_list": state, "next_state_list": next_state,
                "action_list": action, "reward_list": reward,
                "done_list": done}
    def get_transitions(self,start,end):
        batch = self.buffer[start:end]
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return {"state_list": state, "next_state_list": next_state,
                "action_list": action, "reward_list": reward,
                "done_list": done}
        # return state, action, reward, next_state, done
    def get_all_states(self):
        state, _, _, _, _ = map(np.stack, zip(*self.buffer))
        return state

    def convert_D4RL(self, dataset):
        data_size = int(dataset['observations'].shape[0])  # only use maximal 100_000 data points
        for i in range(data_size):
            self.push(dataset['observations'][i],
                      dataset['actions'][i],
                      dataset['rewards'][i],
                      dataset['next_observations'][i],
                      dataset['terminals'][i])
    def distill_D4RL(self, dataset,sample_method="best",ratio=0.05,traj_num = 50):
        # random distill dataset, keep at least 50_000 data points. #据随机提取数据
         # at least keep 50_000 data
        if sample_method == "random":
            data_size = max(int(ratio * dataset['observations'].shape[0]), 50_000)
            index = np.random.randint(0, dataset['observations'].shape[0], size=data_size)
            for i in index:
                self.push(dataset['observations'][i],
                          dataset['actions'][i],
                          dataset['rewards'][i],
                          dataset['next_observations'][i],
                          dataset['terminals'][i])
        elif sample_method == "best":  # select the last data
            data_size = int(dataset['observations'].shape[0])  # only use maximal 100_000 data points
            episode_reward = 0
            queue = PriorityQueue()
            traj = []
            for i in trange(data_size - 1):
                if not dataset['terminals'][i] and not dataset['timeouts'][i] and len(traj) < 1000 :
                    state = dataset['observations'][i]
                    action = dataset['actions'][i]
                    reward = dataset['rewards'][i]
                    done = dataset['terminals'][i]
                    # if dataset['terminals'][i] == True:
                    #     print(11)
                    next_state = dataset['observations'][i + 1]
                    traj.append((state, action, reward, next_state, done))
                    episode_reward += reward
                else:
                    if episode_reward == 0:
                        continue
                    queue.put((1000/episode_reward+ 100, traj))
                    traj = []
                    episode_reward = 0
            for i in range(traj_num):
                traj = queue.get()[1]
                for transition in traj:
                    self.push_transition(transition)
        else:
            raise ValueError
    def __len__(self):
        return len(self.buffer)
