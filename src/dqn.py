import os
import numpy as np
import tensorflow as tf
import math
import utils
from tensorflow.python import debug as tf_debug
import json


class DQN:

    def __init__(self, config):
        self.epochs = config['epochs']
        self.learning_rate = config['learning_rate']
        self.target_update = config['target_update']
        self.gamma = config['gamma']
        self.model_name = config['model_name']
        self.seed = config['seed']
        self.log_step = config['log_step']
        self.train_batch_size = config['train_batch_size']
        self.valid_batch_size = config['valid_batch_size']
        self.optimizer = config['optimizer']
        self.initializer = config['initializer']
        self.export_dir = config['export_dir']
        self.SERVING_DIR = os.path.join(self.export_dir, self.model_name+'_serving', '1')
        self.TF_SUMMARY_DIR = os.path.join(self.export_dir, self.model_name+'_summary')
        self.CKPT_DIR = os.path.join(self.export_dir, self.model_name+'_checkpoint')
        self.split = config['split']
        self.action_dim = config['num_actions']

        self.INITIALIZERS = {
            'xavier': tf.glorot_uniform_initializer(), 
            'uniform': tf.random_uniform_initializer(-1, 1)
        }

        self.OPTIMIZERS = {
            'sgd': tf.train.GradientDescentOptimizer(self.learning_rate),
            'adam': tf.train.AdamOptimizer(self.learning_rate),
            'sgd_mom': tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9, use_nesterov=True),
            'rmsprop': tf.train.RMSPropOptimizer(self.learning_rate),
            'adagrad': tf.train.AdagradOptimizer(self.learning_rate)
        }

        self.LOSSES = {
            'mse': tf.losses.mean_squared_error,
            'huber': tf.losses.huber_loss
        }

        if self.optimizer not in self.OPTIMIZERS.keys():
            raise ValueError("optimizer should be in {}".format(self.OPTIMIZERS.keys()))
        
        if self.export_dir is None:
            raise ValueError("export_dir cannot be empty")

    def input_fn(self, transition_matrices):

        # Fetch current_state, action, reward and next_state matrices.
        current_states, actions, rewards, next_states, end = transition_matrices

        current_states = current_states.astype(np.float32)
        actions = actions.astype(np.float32)
        rewards = rewards.astype(np.float32)
        next_states = next_states.astype(np.float32)
        end = end.astype(np.float32)

        # Convert action dtype for indexing.
        actions = actions.astype(np.int32)

        # Split dataset into train and validation set.
        split_percentage = self.split
        num_samples = len(current_states)
        train_size = int(split_percentage * num_samples)
        valid_size = int((1-split_percentage) * num_samples)
        train_set = (current_states[:train_size], actions[:train_size],
                     rewards[:train_size], next_states[:train_size],
                     end[:train_size])
        valid_set = (current_states[-valid_size:], actions[-valid_size:],
                     rewards[-valid_size:], next_states[-valid_size:],
                     end[:valid_size])

        # Calculate number of train batches.
        self.num_train_batches = int(math.ceil(train_size / float(self.train_batch_size)))
        # Calculate number of valid batches.
        self.num_valid_batches = int(math.ceil(valid_size / float(self.valid_batch_size)))

        # Create Dataset object from input.
        train_dataset = tf.data.Dataset.from_tensor_slices(train_set).batch(self.train_batch_size)
        valid_dataset = tf.data.Dataset.from_tensor_slices(valid_set).batch(self.valid_batch_size)

        # Create generic iterator.
        data_iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

        # Create initialisation operations.
        train_init_op = data_iter.make_initializer(train_dataset)
        valid_init_op = data_iter.make_initializer(valid_dataset)

        return train_init_op, valid_init_op, data_iter

    def q_network(self, states, variable_scope, trainable):
        """Computes the action-value function (Q value) at a given state and
        action"""
        with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):
            q = tf.layers.dense(states, self.action_dim, activation=tf.nn.relu,
                               trainable=trainable)
            return q

    def loss_fn(self, current_states, actions, rewards, next_states, end):
        """Computes the loss to update soft Q function."""
        with tf.name_scope("loss_fn"):

            with tf.name_scope("primary"):
                q_values = self.q_network(current_states,
                                     variable_scope="q_network", trainable=True)
                tf.summary.histogram("q_value", q_values)

                actions_x_idx = tf.reshape(actions, [-1])
                actions_y_idx = tf.range(start=0, limit=tf.shape(actions)[0])
                actions_idx = tf.stack([actions_y_idx, actions_x_idx], axis=1)
                q_primary_ = tf.gather_nd(q_values, actions_idx)
                q_primary = tf.reshape(q_primary_, [-1, 1])


            with tf.name_scope("target"):
                target_q_values = self.q_network(next_states,
                                            variable_scope="target_q_network",
                                           trainable=False)
                tf.summary.histogram("target_q_vallue", target_q_values)
                max_q_target = tf.reshape(tf.reduce_max(target_q_values, axis=1), [-1, 1])

                td_target = tf.add(rewards, self.gamma * max_q_target * end)

            td_error = td_target - q_primary
            # TODO: error term clipping between -1 and 1
            loss_op = tf.reduce_sum(tf.square(td_error))
 
            return loss_op

    def optimize_fn(self, loss_op):
        """
        Optimization function for the Backpropagation. 
        Dervied class can override this function to implement custom changes to optimization.
        
        Parameters
        ----------
            loss: Tensor shape=[1,1]
                Computed loss for all the samples in batch,
                output of `_loss_fn()`.
        
        Returns
        -------
            optimize_op: Tensorflow Op
                Optimization operation to be performed on loss.
        """
        with tf.variable_scope('optimization'):
            # Select the optimizer.
            optimizer = self.OPTIMIZERS[self.optimizer]

            # Minimize loss based on optimizer. 
            optimize_op = optimizer.minimize(loss_op)

            # Calculate gradients using the optimizer and the loss function.
            gradients = optimizer.compute_gradients(loss_op)

            # Apply clipped gradients.
            optimize_op = optimizer.apply_gradients(gradients)

            # Add summaries of gradients to tensorboard.
            utils.gradient_summaries(gradients)

            return optimize_op

    def train(self, current_states, actions, rewards, next_states, end):

        # Create loss operation for Q function update.
        loss_op = self.loss_fn(current_states, actions, rewards,
                                             next_states, end)

        # Create optimization operation.
        optimize_op = self.optimize_fn(loss_op)

        # Log loss in tensorboard summary.
        #mean_loss_op, mean_loss_update_op = utils.avg_loss(loss_op)
        tf.summary.scalar('loss', loss_op)

        # Summaries for all the trainable variables.
        utils.parameter_summaries(tf.trainable_variables())

        # TODO: Add tensorboard model evaluation metrics.

        summary = tf.summary.merge_all()

        return optimize_op, loss_op, summary

    def copy(self, primary_scope, target_scope):

        with tf.name_scope("copy"):

            primary_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=primary_scope)
            target_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_scope)

            primary_variables_sorted = sorted(primary_variables, key=lambda v: v.name)
            target_variables_sorted = sorted(target_variables, key=lambda v: v.name)

            assign_ops = []

            for primary_var, target_var in zip(primary_variables_sorted, target_variables_sorted):
                assign_ops.append(target_var.assign(tf.identity(primary_var)))

            copy_op = tf.group(*assign_ops)

            return copy_op

    def fit(self, transition_matrices, restore=False, global_step=0):

        # Check if the export directory is present,
        # if not present create new directory.
        if os.path.exists(self.export_dir) and restore is False:
            raise ValueError("Export directory already exists. Please specify different export directory.")
        elif os.path.exists(self.export_dir) and restore:
            print ("Restoring model from latest checkpoint.")
            pass
        else:
            os.mkdir(self.export_dir)

        # self.builder=tf.saved_model.builder.SavedModelBuilder(self.SERVING_DIR)

        # Save model config
        params = self.get_params()
        with open(os.path.join(self.export_dir, 'params.json'), 'w') as f:
            json.dump(params, f)


        # Clear deafult graph stack and reset global graph definition.
        tf.reset_default_graph()

        # Set seed for random.
        tf.set_random_seed(self.seed)

        # Get data iterator ops.
        train_init_op, valid_init_op, data_iter = self.input_fn(transition_matrices)

        # Create iterator.
        current_states, actions, rewards, next_states, end = data_iter.get_next()

        # Get loss and optimization ops
        optimize_op, loss_op, summary = self.train(current_states, actions,
                                                   rewards, next_states, end)


        # Create model copy op.
        copy_op = self.copy(primary_scope='q_network',
                            target_scope='target_q_network')

        # Object to saver model checkpoints
        self.saver = tf.train.Saver()
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
        with tf.Session() as sess:

            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            # Create file writer directory to store summary and events.
            graph_writer = tf.summary.FileWriter(self.TF_SUMMARY_DIR+'/graph', sess.graph)
            writer = tf.summary.FileWriter(self.TF_SUMMARY_DIR+'/train')

            # Initialize variables in graph.
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # Restore model checkpoint.
            if restore:
                self.saver.restore(sess, self.CKPT_DIR+"{}.ckpt".format(self.model_name))

            # Initialize step count.
            step = global_step
            for epoch in range(self.epochs):

                # Initialize training set iterator.
                sess.run(train_init_op)

                for batch in range(self.num_train_batches):

                    _, train_loss, train_summary = sess.run([optimize_op,
                                                             loss_op, summary],
                                                           options=run_options)
                    if epoch % self.log_step == 0:
                        print ("training: ", step, train_loss)

                    # Log training dataset.
                    writer.add_summary(train_summary, step)

                    # Check if step to update Q target.
                    if step % self.target_update == 0:
                        print ("copying parameters {}".format(step))
                        sess.run(copy_op)

                    step +=1

                """
                # Log results every step.
                if epoch % self.log_step == 0:

                    # Get validation set.
                    # Initialize training set iterator.
                    sess.run(valid_init_op)

                    # Get results on validation set.
                    valid_loss, valid_summary = sess.run([loss_op, summary])

                    # Log validation dataset.
                    valid_writer.add_summary(valid_summary, step)
                """
                self.saver.save(sess, self.CKPT_DIR+"{}.ckpt".format(self.model_name))

            return step

    def predict(self, test_X):
        # Clear deafult graph stack and reset global graph definition.
        tf.reset_default_graph()

        # Get data iterator ops.
        # _, _, data_iter = self.input_fn(transition_matrices)

        # Create iterator.
        # current_states, _, _, _ = data_iter.get_next()
        current_states = tf.placeholder(shape=[None, 4], dtype=tf.float32)

        q_values = self.q_network(current_states,variable_scope="q_network",trainable=True)
        action = tf.argmax(q_values, axis=1)

        # Object to saver model checkpoints
        self.saver = tf.train.Saver()
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
        with tf.Session() as sess:
            # Restore model checkpoint.
            self.saver.restore(sess, self.CKPT_DIR+"{}.ckpt".format(self.model_name))

            # Result on test set batch.
            action_test = sess.run([action], {current_states:
                                              test_X.reshape(-1, 4)},
                                   options=run_options)
        return action_test[0]

    def get_params(self, verbose=False):
        """
        Arguments with values of RNN.

        Parameters
        ----------
            verbose: boolean
                If True, returns all the class variables.

        Returns
        -------
            params: dictionary
                A dictionary of parameters in the model and their values.
        """
        params = dict()
        exclude = set(('builder', 'MODELS', 'INITIALIZERS', 'OPTIMIZERS', 'LOSSES', 
                       'CKPT_DIR', 'TF_SUMMARY_DIR', 'SERVING_DIR', 
                       'prediction_signature', 'saver', 'export_dir'))
        keys = self.__dict__.keys()

        if verbose == False:
            keys = list(set(keys) - exclude)

        for param in keys:
            params[param] = getattr(self, param)

        return params
