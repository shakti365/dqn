import numpy as np
from collections import defaultdict


class ReplayMemory:

    def __init__(self, capacity):
        self.current_episode_id = 0
        self.step_id = 0
        self.capacity = capacity

        self.episodes = np.empty((0, 1), dtype=np.float32)
        self.steps = np.empty((0, 1), dtype=np.float32)
        self.states = np.empty((0, 4), dtype=np.float32)
        self.actions = np.empty((0, 1), dtype=np.float32)
        self.rewards = np.empty((0, 1), dtype=np.float32)
        self.end = np.empty((0, 1), dtype=np.float32)
        self.next_states = np.empty((0, 4), dtype=np.float32)

    def add_sample(self, episode_id, mdp):
        # create a local memory for each episode with metadata information
        self.step_id += 1
        state, action, reward, next_state, end = mdp['current_state'], mdp['action'], mdp['reward'], mdp['next_state'], mdp['end']

        if type(action) is not list:
            action = np.array(action)

        self.episodes = np.append(self.episodes[-self.capacity:],
                                  np.array(self.current_episode_id, ndmin=2), axis=0)
        self.steps = np.append(self.steps[-self.capacity:],
                               np.array(self.step_id, ndmin=2), axis=0)
        self.states = np.append(self.states[-self.capacity:],
                                state.reshape(-1, 4), axis=0)
        self.actions = np.append(self.actions[-self.capacity:],
                                 action.reshape(-1, 1), axis=0)
        self.rewards = np.append(self.rewards[-self.capacity:],
                                 np.array(reward, ndmin=2), axis=0)
        self.next_states = np.append(self.next_states[-self.capacity:],
                                     next_state.reshape(-1, 4), axis=0)
        self.end = np.append(self.end[-self.capacity:], np.array(end, ndmin=2), axis=0)

    def fetch_sample(self, num_samples=None):

        if num_samples is None:
            num_samples = len(self.states)

        idx = np.random.choice(range(len(self.states)), size=num_samples, replace=False)
        states = self.states[idx]
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        next_states = self.next_states[idx]
        end = self.end[idx]
        return states, actions, rewards, next_states, end

    def save_memory(self):
        pass

    def create_episode(self):
        self.current_episode_id += 1
        self.step_id = 0
        return self.current_episode_id
