import gym
from replay_memory import ReplayMemory
from collections import deque


class Agent:

    def __init__(self, render=False, model=None, memory_capacity=1000000,
                 state_buffer_size=4):
        # create an environment
        self.environment = gym.make('MountainCarContinuous-v0')
        # reset environment when an agent is initialized
        self.current_observation = self.reset_environment()
        self.render = render
        self.model = model
        # Create a replay memory for the agent.
        self.memory = ReplayMemory(capacity=memory_capacity)
        self.state_buffer = deque(list())
        self.state_buffer_size = state_buffer_size

    def reset_environment(self):
        current_observation = self.environment.reset()
        return current_observation

    def get_action(self, current_state):
        """Fetch an action according to model policy"""
        if self.model is None:
            action = self.environment.action_space.sample()
        else:
            action = self.model.predict(current_state)
        return action

    def process_observations(self, observations):
        """Processes an input of observations and returns the state"""
        # silly processing to return just the last observation
        # TODO: understand how it changes the training
        return observations[-1]

    def get_state(self, observation):
        """Add current observation to state buffer and process this observation
        to convert it into state"""
        if len(self.state_buffer) > self.state_buffer_size:
            self.state_buffer.popleft()

        self.state_buffer.append(observation)
        state = self.process_observations(self.state_buffer)
        return state

    def get_transitions(self, action):
        """Take one step in the environment and return the observations"""
        next_observation, reward, done, _ = self.environment.step(action)
        if self.render:
            self.environment.render()
        next_state = self.get_state(next_observation)
        return next_state, reward, done

    def run_episode(self, num_episodes=1):
        """run episodes `num_episodes` times using `model` policy"""
        for episode in range(num_episodes):
            self.current_observation = self.reset_environment()
            episode_id = self.memory.create_episode()

            done = False
            transition = dict()

            while not done:
                transition['current_state'] = self.get_state(self.current_observation)
                transition['action'] = self.get_action(self.current_observation)
                transition['next_state'], transition['reward'], done = self.get_transitions(transition['action'])

                self.memory.add_sample(episode_id, transition)

            self.memory.add_episode(episode_id)

    def learn(self, step=0, restore=False):
        """Train model using transitions in replay memory"""
        if self.model is None:
            raise Exception("This agent has no brain! Add a model which implements fit() function to train.")

        # Sample array of transitions from replay memory.
        transition_matrices = self.memory.fetch_sample()

        if step != 0:
            restore = True

        # Fit the SAC model.
        self.model.fit(transition_matrices, restore=restore, global_step=step)
