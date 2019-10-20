import os
from agent import Agent
from dqn import DQN
from easy_tf_log import tflog
import random
from datetime import datetime
import pytz
import tensorflow as tf
import numpy as np
import time

from logger import Logger
logger = Logger()

logger.log(input())

np.random.seed(0)

# uid = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d_%H:%M:%S")
model_name = "exp-1-2019-10-20_18:13:42" # "exp-1-2019-10-16_00:18:36" "exp-1-2019-10-16_00:04:51"
data_dir = os.path.join(os.pardir, 'data', 'models')
export_dir = os.path.join(data_dir, model_name)

# Initialize replay memory to capacity
render = True
num_dummy_episodes=100
memory_capacity = 5000
agent = Agent(render=render, model=None, memory_capacity=memory_capacity)
# agent.run_episode(num_episodes=num_dummy_episodes)

config = dict()
config['epochs'] = 1
config['learning_rate'] = 0.001
config['target_update'] = 1000
config['gamma'] = 0.9
config['model_name'] = model_name
config['seed'] = 42
config['log_step'] = 5000
config['train_batch_size'] = 64
config['valid_batch_size'] = 64
config['optimizer'] = 'adam'
config['initializer'] = 'xavier'
config['export_dir'] = export_dir
config['split'] = 0.9
config['num_actions'] = 2

dqn = DQN(config)
agent.model = dqn

num_episodes = 5000
#num_steps = 10
num_samples = 256
epsilon = np.logspace(start=0, stop=5, base=0.05, num=num_episodes)

global_step = 0
total_reward = 0
exploration = 0
exploitation = 0

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


episode_rewards = []
average_reward = 0
previous_best = 0

with tf.Session() as sess:

    saver.restore(sess, dqn.CKPT_DIR+"{}.ckpt".format(dqn.model_name))

    # Create writer to store summaru.
    writer = tf.summary.FileWriter(dqn.TF_SUMMARY_DIR+'/train', sess.graph)

    # Initialize variables in graph.
    # sess.run(tf.global_variables_initializer())
    # sess.run(tf.local_variables_initializer())

    # For episode:
    for episode in range(num_episodes):

        total_episode_reward = 0

        actions_counter = {
            0: 0,
            1: 0,
        }

        # Observe current state
        agent.current_observation = agent.reset_environment()

        done = False
        # For step:
        while not done:
            transition = dict()
            transition['current_state'] = agent.get_state(agent.current_observation)

            current_epsilon = epsilon[episode]
            # Epsilon greedy policy
            # if (random.uniform(0, 1) < current_epsilon) or (global_step == 0):
            #     # Get a random action.
            #     transition['action'] = agent.get_action(agent.current_observation,
            #                                            random=True)
            #     exploration += 1
            # else:
            # Get recommended action from the policy
            transition['action'], q_values_ = sess.run([action, q_values], {current_states:
                                                    transition['current_state'].reshape(-1,4)})
            #logger.log(q_values_)
            exploitation += 1


            # Observe next state x and reward r
            transition['next_state'], transition['reward'], done = agent.get_transitions(transition['action'])
            # if done is True:
            #     transition['end'] = 0.0
            # else:
            #     transition['end'] = 1.0

            # Push transition to replay memory
            # agent.memory.add_sample(episode, transition)

            # transition_matrices = agent.memory.fetch_sample(num_samples)

            # feed_dict ={
            #     current_states: transition_matrices[0].astype(np.float32),
            #     actions: transition_matrices[1].astype(np.int32),
            #     rewards: transition_matrices[2].astype(np.float32),
            #     next_states: transition_matrices[3].astype(np.float32),
            #     end: transition_matrices[4].astype(np.float32)
            # }

            # train_loss, _, train_summary = sess.run([loss_op, optimize_op, summary], feed_dict)

            # if global_step % dqn.log_step == 0:
            #     #logger.log(f"training: {global_step}, {train_loss}")
            #     # Log training dataset.
            #     writer.add_summary(train_summary, global_step)
                # agent.log_rewards(total_episode_reward, step=global_step)


            # Check if step to update Q target.
            # if global_step % dqn.target_update == 0:
            #     logger.log(f"copying parameters {global_step}")
            #     sess.run(copy_op)

            # global_step += 1
            total_episode_reward += transition['reward']
            # actions_counter[transition['action'][0] if type(transition['action']) is not int else transition['action']] += 1

        episode_rewards.append(total_episode_reward)
        # average_reward = sum(episode_rewards[-100:]) / 100.0

        print(f"episode: {episode} reward: {total_episode_reward}")
        # logger.log(f"Step: {global_step},  Current Epsilon: {current_epsilon}")
        # logger.log(f"Step: {global_step},  Exploration: {exploration/global_step}")
        # logger.log(f"Step: {global_step},  Exploitation: {exploitation/global_step}")
        # logger.log(f"Actions: {actions_counter}")
        

        # if episode % 100 == 0:
        #     average_reward = sum(episode_rewards[-100:]) / 100.0
        #     agent.log_avg_rewards(total_episode_reward, global_step)

        #     if average_reward > previous_best:
        #         logger.log(f"saving model at {average_reward} > {previous_best}")
        #         previous_best = average_reward
        #         saver.save(sess, dqn.CKPT_DIR+"{}.ckpt".format(dqn.model_name))

        # if average_reward >= 150:
        #     break

            # Observe current state
            # agent.render = True
            # for ep in range(10):
            #     total_episode_reward = 0

            #     test_current_observation = agent.reset_environment()

            #     test_done = False
            #     # For step:
            #     while not test_done:
            #         test_current_state = agent.get_state(test_current_observation)
            #         # Get recommended action from the policy
            #         test_action, _ = sess.run([action, q_values], {current_states:
            #                                               test_current_state.reshape(-1,4)})

            #         test_next_state, test_reward, test_done = agent.get_transitions(test_action)

            #         time.sleep(0.1)

            #         total_episode_reward += test_reward

            #     print("test: ", total_episode_reward)
            #     agent.log_avg_rewards(total_episode_reward, global_step)

            # agent.render = False
