import os
from agent import Agent
from dqn import DQN
from easy_tf_log import tflog
import random
from datetime import datetime
import pytz
import tensorflow as tf
import numpy as np

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

num_episodes = 100
num_steps = 10
num_samples = 32
epsilon = 0.5

global_step = 0
total_reward = 0


# Clear deafult graph stack and reset global graph definition.
tf.reset_default_graph()

# Set seed for random.
tf.set_random_seed(dqn.seed)

# Create input placeholders.
current_states = tf.placeholder(shape=[None, 4], dtype=tf.float32)
actions = tf.placeholder(shape=[None, 1], dtype=tf.int32)
rewards = tf.placeholder(shape=[None, 1], dtype=tf.float32)
next_states = tf.placeholder(shape=[None, 4], dtype=tf.float32)
end = tf.placeholder(shape=[None, 1], dtype=tf.float32)

q_values = dqn.q_network(current_states, variable_scope='q_network',
                                trainable=True)

action = tf.argmax(q_values, axis=1)

# Get loss and optimization ops
optimize_op, loss_op, summary = dqn.train(current_states, actions,
                                           rewards, next_states, end)


# Create model copy op.
copy_op = dqn.copy(primary_scope='q_network',
                    target_scope='target_q_network')

saver = tf.train.Saver()

with tf.Session() as sess:

    # Create writer to store summaru.
    writer = tf.summary.FileWriter(dqn.TF_SUMMARY_DIR+'/train', sess.graph)

    # Initialize variables in graph.
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

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
                transition['action'] = sess.run(action, {current_states:
                                                      transition['current_state'].reshape(-1,4)})

            # Observe next state x and reward r
            transition['next_state'], transition['reward'], done = agent.get_transitions(transition['action'])
            if done is True:
                transition['end'] = 1.0
            else:
                transition['end'] = 0.0

            # Push transition to replay memory
            agent.memory.add_sample(episode, transition)

            transition_matrices = agent.memory.fetch_sample(num_samples)

            feed_dict ={
                current_states: transition_matrices[0].astype(np.float32),
                actions: transition_matrices[1].astype(np.int32),
                rewards: transition_matrices[2].astype(np.float32),
                next_states: transition_matrices[3].astype(np.float32),
                end: transition_matrices[4].astype(np.float32)
            }

            train_loss, _, train_summary = sess.run([loss_op, optimize_op, summary], feed_dict)

            if global_step % dqn.log_step == 0:
                print ("training: ", global_step, train_loss)

            # Log training dataset.
            writer.add_summary(train_summary, global_step)

            # Check if step to update Q target.
            if global_step % dqn.target_update == 0:
                print ("copying parameters {}".format(global_step))
                sess.run(copy_op)
            """
            if global_step == 0:
                agent.model = dqn

            if episode == 0 and global_step == 0:
                agent.learn(step=global_step, restore=False, sample_size=num_samples)
            else:
                agent.learn(step=global_step, restore=True, sample_size=num_samples)
            """
            global_step += 1
            total_episode_reward += transition['reward']

        print ("episode: {} reward: {}".format(episode, total_episode_reward))
        #agent.log_rewards(total_episode_reward, step=episode)
