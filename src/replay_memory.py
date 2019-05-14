import numpy as np
from collections import defaultdict


class ReplayMemory:

    def __init__(self, capacity):
        self.current_episode_id = 0
        self.step_id = 0

        self.episodes = np.empty((0, 1), dtype=np.float32)
        self.steps = np.empty((0, 1), dtype=np.float32)
        self.states = np.empty((0, 2), dtype=np.float32)
        self.actions = np.empty((0, 1), dtype=np.float32)
        self.rewards = np.empty((0, 1), dtype=np.float32)
        self.next_states = np.empty((0, 2), dtype=np.float32)
        self.replay_memory = dict()


    def add_episode(self, episode_id):
        # Push from local memory with an id to replay memory
        mdp = self.replay_memory[episode_id]
        for step in range(1, len(mdp)):

            state, action, reward, next_state = mdp[step]['current_state'], mdp[step]['action'], mdp[step]['reward'], mdp[step]['next_state']

            self.episodes = np.append(self.episodes, np.array(episode_id, ndmin=2), axis=0)
            self.steps = np.append(self.steps, np.array(step, ndmin=2), axis=0)
            self.states = np.append(self.states, state.reshape(-1, 2), axis=0)
            self.actions = np.append(self.actions, action.reshape(-1, 1), axis=0)
            self.rewards = np.append(self.rewards, np.array(reward, ndmin=2), axis=0)
            self.next_states = np.append(self.next_states, next_state.reshape(-1, 2), axis=0)

    def add_sample(self, episode_id, transition):
        # create a local memory for each episode with metadata information
        step = len(self.replay_memory[episode_id])
        self.replay_memory[episode_id][step] = transition
    """
    def terminate_episode(self):
        # Increment ID counters
        self.current_episode_id += 1
        self.step_id += 1

        # Reset local memory
        self.episodes = []
        self.steps = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
    """

    def fetch_sample(self, num_samples=None):

        if num_samples is None:
            num_samples = len(self.states)

        idx = np.random.choice(range(len(self.states)), size=num_samples, replace=False)
        states = self.states[idx]
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        next_states = self.next_states[idx]
        return states, actions, rewards, next_states

    def save_memory(self):
        pass

    def create_episode(self):
        self.current_episode_id += 1
        self.replay_memory[self.current_episode_id] = defaultdict(list)
        return self.current_episode_id
