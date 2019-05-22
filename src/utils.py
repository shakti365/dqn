from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)

import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, roc_auc_score, f1_score, accuracy_score

np.random.seed(42)

def avg_loss(loss):
    with tf.variable_scope("metrics"):
        # create variable for storing sum and count of loss to calculate mean over batch. 
        total = tf.get_variable(initializer=0.0, dtype=tf.float32, name='total_loss', trainable=False)
        count = tf.get_variable(initializer=0.0, dtype=tf.float32, name='count_loss', trainable=False)

        with tf.name_scope('mean_loss'):
            update_total = tf.assign_add(total, loss)
            update_count = tf.assign_add(count, 1.0)
            mean_loss = tf.divide(total, count)
            mean_loss_update = tf.group([update_total, update_count])

    return mean_loss, mean_loss_update

def preprocess_image(image):
    """
    Preprocess the RGB image of observation for neural net input.

    Params:
    -------
        image: numpy [height, width, channels]
            A 3D matrix of RGB pixel values.

    Returns:
    --------
        processed_image: numpy [80*80]
            A 1D numpy array containing binary values, 1 for padels and ball else 0.
    """

    # Crop out relevant part of the image
    image = image[35:195]

    # Downsampling the image, taking only alternative pixels to reduce the size of input further.
    image = image[::2, ::2, 0]

    # Remove backgrounds and convert image to binary pixel values.
    image[image == 144] = 0
    image[image == 109] = 0
    image[image != 0] = 1

    # Flatten image.
    processed_image = image.astype(np.float32).ravel()

    return processed_image

def variable_summaries(var, var_name=None):
    """
    Creates summaries of a tensor variable.

    Parameters
    ----------
        var: Tensor
             A tensor variable for which the summaries have to be created.
        var_name: String (default None)
             Name with which the summaries are logged.
             If None, it takes the name of the variable.
    """
    if var_name is None:
        var_name= var.name

    mean = tf.reduce_mean(var)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('{}_mean'.format(var_name), mean)
    tf.summary.scalar('{}_stddev'.format(var_name), stddev)
    tf.summary.histogram('{}_histogram'.format(var_name), var)

def gradient_summaries(gradients):
    """
    Creates gradient summaries.

    Parameters
    ----------
        gradients: List of Tuple [(Tensor,Tensor)]
            A list of tuples of tensor varible and values.
    """
    with tf.name_scope('gradients'):
        # Save the gradients summary for tensorboard.
        for grad in gradients:
            # Assign a name to identify gradients.
            var_name = '{}-grad'.format(grad[1].name)
            if 'bias' not in var_name:
                variable_summaries(grad[0], var_name=var_name)

def parameter_summaries(variables):
    """
    Creates parameter summaries.

    Parameter
    ---------
        variables: List of Tensor
            A list of trainable parameters in the model.
    """
    with tf.name_scope('variables'):
        # Save the gradients summary for tensorboard.
        for var in variables:
            # Assign a name to identify gradients.
            var_name = '{}-variable'.format(var.name)
            if 'bias' not in var_name:
                variable_summaries(var, var_name=var_name)

