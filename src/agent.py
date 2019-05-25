import gym
from replay_memory import ReplayMemory
from collections import deque
import tensorflow as tf


class Agent:

    def __init__(self, render=False, model=None, memory_capacity=1000000,
                 state_buffer_size=4):
        # create an environment
        self.environment = gym.make('CartPole-v0')
        # reset environment when an agent is initialized
        self.current_observation = self.reset_environment()
        self.render = render
        self.model = model
        # Create a replay memory for the agent.
        self.memory = ReplayMemory(capacity=memory_capacity)
        self.state_buffer = deque(list())
        self.state_buffer_size = state_buffer_size

        # Logging variables initialization
        self.episode_count = 0
        self.total_reward = 0

    def reset_environment(self):
        current_observation = self.environment.reset()
        return current_observation

    def get_action(self, current_state, random=False):
        """Fetch an action according to model policy"""
        if self.model is None or random is True:
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
        action = int(action)
        next_observation, reward, done, _ = self.environment.step(action)
        if self.render:
            self.environment.render()
        next_state = self.get_state(next_observation)
        self.current_observation = next_observation

        return next_state, reward, done

    def run_episode(self, num_episodes=1):
        """run episodes `num_episodes` times using `model` policy"""
        for episode in range(num_episodes):

            if len(self.memory.states) > self.memory.capacity:
                print ("replay memory full to its capacity {}".format(self.memory.capacity))
                break

            self.current_observation = self.reset_environment()
            episode_id = self.memory.create_episode()

            done = False
            transition = dict()

            while not done:
                transition['current_state'] = self.get_state(self.current_observation)
                transition['action'] = self.get_action(self.current_observation)
                transition['next_state'], transition['reward'], done = self.get_transitions(transition['action'])
                if done is True:
                    transition['end'] = 1
                else:
                    transition['end'] = 0

                self.memory.add_sample(episode_id, transition)

    def learn(self, step=0, restore=False, sample_size=1):
        """Train model using transitions in replay memory"""
        if self.model is None:
            raise Exception("This agent has no brain! Add a model which implements fit() function to train.")

        # Sample array of transitions from replay memory.
        transition_matrices = self.memory.fetch_sample(sample_size)

        if step != 0:
            restore = True

        # Fit the SAC model.
        self.model.fit(transition_matrices, restore=restore, global_step=step)

    def log_rewards(self, episode_total_reward, step):
        writer = tf.summary.FileWriter(self.model.TF_SUMMARY_DIR+'/reward')
        summary = tf.Summary(value=[tf.Summary.Value(tag="average_reward",
                                                     simple_value=episode_total_reward)])
        writer.add_summary(summary, step)
