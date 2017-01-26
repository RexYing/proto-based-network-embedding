from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

ENCODER_NUM_HIDDEN_NEURONS = [100, 50]
DECODER_NUM_HIDDEN_NEURONS = [100]

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
                                      
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape,
    initializer=initializer, dtype=dtype)
  return var




def encode(X):
  batch_size = X.get_shape()[0]
  input_len = X.get_shape()[1]
  
  encoding_weights = [_variable_on_cpu('weights1', [input_len, ENCODER_NUM_HIDDEN_NEURONS[0]],
                                       tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)),
                      _variable_on_cpu('weights2', [ENCODER_NUM_HIDDEN_NEURONS[0], ENCODER_NUM_HIDDEN_NEURONS[1]],
                                       tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32))]

  with tf.variable_scope('encode1') as scope:
    b1 = _variable_on_cpu('biases', [ENCODER_NUM_HIDDEN_NEURONS[0]], tf.constant_initializer(0.0))
    h1 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(X, encoding_weights[0]), b1), name=scope.name + '/hidden1')

  with tf.variable_scope('encode2') as scope:
    b2 = _variable_on_cpu('biases', [ENCODER_NUM_HIDDEN_NEURONS[1]], tf.constant_initializer(0.0))
    h2 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(h1, encoding_weights[1]), b2), name=scope.name + '/hidden2')
  
  return h2, encoding_weights

def decode(h, input_len, encoding_weights = None):

  if not encoding_weights:
    encoding_weights = [_variable_on_cpu('weights1_decode', [input_len, ENCODER_NUM_HIDDEN_NEURONS[0]],
                                         tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)),
                        _variable_on_cpu('weights2_decode', [ENCODER_NUM_HIDDEN_NEURONS[0], ENCODER_NUM_HIDDEN_NEURONS[1]],
                                         tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32))]
  
  with tf.variable_scope('decode1') as scope:
    b1 = _variable_on_cpu('bias', [DECODER_NUM_HIDDEN_NEURONS[0]], tf.constant_initializer(0.0))
    h1 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(h, tf.transpose(encoding_weights[1])), b1), name=scope.name + '/hidden3')

  with tf.variable_scope('decode2') as scope:
    b2 = _variable_on_cpu('bias', [input_len], tf.constant_initializer(0.0))
    h2 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(h1, tf.transpose(encoding_weights[0])), b2), name = scope.name +
                    '/hidden4')

  return h2

def loss(inferred, labels):
  """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".

  Args:
    inferred_values: Sensor values output from inference().
    labels: Labels. 1-D tensor of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  #loss_weights = _variable_on_cpu('loss_weights', shape=[FLAGS.receptive_field_size], 
  #                                tf.constant_initializer(1 / FLAGS.receptive_field_size))
  #l2_loss = tf.einsum('i,i->', loss_weights, (tf.nn.l2_loss(tf.sub(inferred, labels)), name='raw_loss'), name='weighted_loss')
  l2_loss = tf.nn.l2_loss(tf.sub(inferred, labels), name='raw_loss')
  tf.add_to_collection('losses', l2_loss)
  tf.summary.scalar(l2_loss.name, l2_loss)
  return l2_loss

def _add_loss_summaries(loss):
  """Add summaries for losses in CIFAR-10 model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns: 
    loss_averages_op: op for generating moving averages of losses.  
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(loss, global_step):
  
  #loss_averages_op = _add_loss_summaries(loss)

  #with tf.control_dependencies([loss_averages_op]):
  opt = tf.train.AdamOptimizer(epsilon=1e-3)
  grads = opt.compute_gradients(loss)

  apply_grad_op = opt.apply_gradients(grads, global_step=global_step)

  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  variable_averages = tf.train.ExponentialMovingAverage(0.9, global_step)
  variable_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_grad_op, variable_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op

