from agent import Agent
from dqn import DQN

# Initialize replay memory to capcity N
render = False
memory_capacity = 10000
agent = Agent(render=render, model=None, memory_capacity=memory_capacity)
agent.run_episode(num_episodes=100)

num_episodes = 1
num_steps = 10
num_samples = 32

config = dict()
config['epochs'] = 1
config['learning_rate'] = 0.001
config['target_update'] = 10
config['gamma'] = 0.9
config['model_name'] = 'test'
config['seed'] = 42
config['log_step'] = 1
config['train_batch_size'] = 32
config['valid_batch_size'] = 32
config['optimizer'] = 'rmsprop'
config['initializer'] = 'uniform'
config['logs_path'] = '../data/models'
config['split'] = 0.9
config['num_actions'] = 2

dqn = DQN(config)

global_step = 0
done = False

# For episode:
for episode in range(num_episodes):

    # Observe current state
    agent.current_observation = agent.reset_environment()

    # For step:
    while not done:
        transition = dict()
        transition['current_state'] = agent.get_state(agent.current_observation)
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

        agent.learn(step=global_step, sample_size=num_samples)
        global_step += 1
