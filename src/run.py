import os
from agent import Agent
from dqn import DQN
from easy_tf_log import tflog
import random
from datetime import datetime
import pytz

uid = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d_%H:%M:%S")
model_name = 'exp-1-{}'.format(uid)
data_dir = os.path.join(os.pardir, 'data', 'models')
export_dir = os.path.join(data_dir, model_name)

# Initialize replay memory to capcity N
render = False
memory_capacity = 10000
agent = Agent(render=render, model=None, memory_capacity=memory_capacity)
agent.run_episode(num_episodes=50)

config = dict()
config['epochs'] = 1
config['learning_rate'] = 0.001
config['target_update'] = 100
config['gamma'] = 0.9
config['model_name'] = model_name
config['seed'] = 42
config['log_step'] = 1
config['train_batch_size'] = 32
config['valid_batch_size'] = 32
config['optimizer'] = 'rmsprop'
config['initializer'] = 'uniform'
config['export_dir'] = export_dir
config['split'] = 0.9
config['num_actions'] = 2

dqn = DQN(config)

num_episodes = 10
num_steps = 10
num_samples = 32
epsilon = 0.5

global_step = 0
total_reward = 0

# For episode:
for episode in range(num_episodes):

    total_episode_reward = 0

    # Observe current state
    agent.current_observation = agent.reset_environment()

    done = False
    # For step:
    while not done:
        transition = dict()
        transition['current_state'] = agent.get_state(agent.current_observation)

        # Epsilon greedy policy
        if (random.uniform(0, 1) < epsilon) or (global_step == 0):
            # Get a random action.
            transition['action'] = agent.get_action(agent.current_observation,
                                                   random=True)
        else:
            # Get recommended action from the policy
            transition['action'] = agent.get_action(agent.current_observation)

        # Observe next state x and reward r
        transition['next_state'], transition['reward'], done = agent.get_transitions(transition['action'])
        if done is True:
            transition['end'] = 1
        else:
            transition['end'] = 0

        # Push transition to replay memory
        agent.memory.add_sample(episode, transition)

        if global_step == 0:
            agent.model = dqn

        if episode == 0 and global_step == 0:
            agent.learn(step=global_step, restore=False, sample_size=num_samples)
        else:
            agent.learn(step=global_step, restore=True, sample_size=num_samples)
        global_step += 1
        total_episode_reward += transition['reward']

    print ("episode: {} reward: {}".format(episode, total_episode_reward))
    agent.log_rewards(total_episode_reward, step=episode)
