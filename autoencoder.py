from __future__ import division, print_function, absolute_import
from models import Model
from layers import Dense
from inits import glorot

import numpy as np
import os
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

LAYER_SIZE = [50, 20, 10]
INIT_LR = 1e-3


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
    tf.summary.scalar(l.op.name +' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op

def create_encoding_weights(layer_size, name_prefix):
  encoding_weights = []
  prev_size = layer_size[0]
  for i in range(1, len(layer_size)):
    curr_size = layer_size[i]
    name = name_prefix + str(i)
    encoding_weights.append(_variable_on_cpu(name, (prev_size, curr_size), 
      tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)))
    prev_size = curr_size
  return encoding_weights


class Autoencoder(Model):
  def __init__(self, placeholders, layer_sizes, tied_weights=True, **kwargs):
    """
      Simple autoencoder constructor.

    Args:
      placeholders: contain placeholders corresponding to 'features' (same as labels), 'dropout'
          (drop probability), 'weight_decay' (decay weights for regularization).
      layer_sizes: the feature dimensions for input, hidden layers and output.
    """
    super(Autoencoder, self).__init__(**kwargs)

    self.inputs = placeholders['features']
    self.input_dim = layer_sizes[0]
    self.output_dim = self.input_dim
    self.placeholders = placeholders
    self.layer_sizes = layer_sizes
    self.embedding_layer_idx = (len(layer_sizes) - 1) // 2
    self.global_step = tf.Variable(0, trainable=False)
    self.logging = True

    self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    self.build()

  def _loss(self, h_neighbors=None, edge_loss_multiplier=0):
      # Weight decay loss
      #for var in self.layers[0].vars.values():
      #    self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

      # Cross entropy error
      #self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
      #                                          self.placeholders['labels_mask'])
    """Add L2Loss to all the trainable variables.
      Add summary for "Loss" and "Loss/avg".
      Since this is an autoencoder, inferred is the same as labels (except DAE).
      If h_neighbors is not None, an l2 loss between hidden vectors for adjacent nodes is added.
  
    Args:
      h_neighbors: hidden representation of neighbors (None if no edge constraint should be
          considered), with dimension (batch size x degree x hidden vector size)
      edge_loss_multiplier: controls how much edge constraint contributes to loss.
    Returns:
      Loss tensor of type float.  
    """
    diff = tf.sub(self.outputs, self.inputs, 'recon_diff')
    l2_loss = tf.reduce_mean(tf.square(diff), name='raw_loss')
    tf.add_to_collection('losses', l2_loss)
  
    # regularization
    gradf = tf.gradients(self.activations[self.embedding_layer_idx], self.inputs)
    gradf_norm = tf.reduce_mean(tf.square(gradf), name='gradnorm')
    reg = tf.mul(self.placeholders['weight_decay'], gradf_norm)
    tf.add_to_collection('losses', reg)
    # l2 regularization
    #loss_weights = _variable_on_cpu('loss_weights', shape=[FLAGS.receptive_field_size], 
    #                                tf.constant_initializer(1 / FLAGS.receptive_field_size))
    #l2_loss = tf.einsum('i,i->', loss_weights, (tf.nn.l2_loss(diff), name='raw_loss'), name='weighted_loss')
  
    # edge constraints
    #y_tiled = tf.tile(inferred, 
  
    tf.summary.scalar(l2_loss.name, l2_loss)
    self.loss = l2_loss
    return l2_loss

  def _accuracy(self):
    self.accuracy = []
    #TODO: measure test accuracy

  def _build(self):

    weights1 = glorot([self.input_dim, self.layer_sizes[1]], name='weights')

    self.layers.append(Dense(input_dim=self.input_dim,
                             output_dim=self.layer_sizes[1],
                             placeholders=self.placeholders,
                             act=tf.nn.relu,
                             dropout=False,
                             logging=self.logging))

    weights1_decode = tf.transpose(weights1)
    self.layers.append(Dense(input_dim=self.layer_sizes[1],
                             output_dim=self.output_dim,
                             placeholders=self.placeholders,
                             act=lambda x: x,
                             dropout=False,
                             shared_weights=weights1_decode,
                             logging=self.logging))

  def _train(self):
    """ Train autoencoder
  
    Returns:
      train_op: op for training.
    """
    
    #loss_averages_op = _add_loss_summaries(self.loss)
  
    #with tf.control_dependencies([loss_averages_op]):
    lr = tf.train.exponential_decay(FLAGS.learning_rate, self.global_step, 1e4, 0.9, name='lr')
    opt = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-2)
    grads = opt.compute_gradients(self.loss)
    apply_grad_op = opt.apply_gradients(grads, global_step=self.global_step)
  
    for grad, var in grads:
      if grad is not None:
        tf.summary.histogram(var.op.name + '/gradients', grad)
  
    variable_averages = tf.train.ExponentialMovingAverage(0.9, self.global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_grad_op, variable_averages_op]):
      train_op = tf.no_op(name='train')
  
    return train_op




def encode(X):
  batch_size = X.get_shape()[0]
  input_len = X.get_shape()[1]
  
  encoding_weights = create_encoding_weights(LAYER_SIZE, 'weights')

  encoding_biases = [_variable_on_cpu('biases1', (LAYER_SIZE[1],),
                                      tf.constant_initializer(0.0)),
                     _variable_on_cpu('biases2', (LAYER_SIZE[2],), 
                                      tf.constant_initializer(0.0))]
  with tf.variable_scope('encode1') as scope:
    h1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(X, encoding_weights[0]), encoding_biases[0]), name=scope.name + '/hidden1')
    #h1 = tf.nn.bias_add(tf.matmul(X, encoding_weights[0]), encoding_biases[0])
    _activation_summary(h1)

  with tf.variable_scope('encode2') as scope:
    h2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(h1, encoding_weights[1]), encoding_biases[1]),
        name=scope.name + '/hidden2')
    _activation_summary(h2)

  gradf = tf.gradients(h2, X)
  
  return h2, encoding_weights, encoding_biases, gradf

def decode(h, input_len, encoding_weights = None):

  if not encoding_weights:
    encoding_weights = create_encoding_weights(LAYER_SIZE, 'decoding_weights')
  
  with tf.variable_scope('decode1') as scope:
    b1 = _variable_on_cpu('biases1', [LAYER_SIZE[1]], tf.constant_initializer(0.0))
    h1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(h, tf.transpose(encoding_weights[1])), b1), name=scope.name + '/hidden3')
    #h1 = tf.nn.bias_add(tf.matmul(h, tf.transpose(encoding_weights[1])), b1)
    _activation_summary(h1)

  with tf.variable_scope('decode2') as scope:
    b2 = _variable_on_cpu('biases2', [input_len], tf.constant_initializer(0.0))
    h2 = tf.nn.bias_add(tf.matmul(h1, tf.transpose(encoding_weights[0])), b2, name = scope.name + '/hidden4')
    _activation_summary(h2)

  return h2

def encode_neighbors(neighbors, encoding_weights, encoding_biases):
  """Encode neighbors

  Args:
    neighbors: 3-D tensor. Dimensions are batch size, degree, feature length neighbors.

  Returns:
    3-D tensor of encoded representation of neighbors. Dimensions are batch size, degree, hiddeen
    feature length.
  """
  W0tiled = tf.tile(encoding_weights[0], tf.pack([tf.shape(neighbors)[0], 1, 1]))
  W1tiled = tf.tile(encoding_weights[1], tf.pack([tf.shape(neighbors)[0], 1, 1]))
  
  with tf.variable_scope('encode_neighbors1') as scope:
    h1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(neighbors, W0tiled), encoding_biases[0]),
                    name=scope.name + '/neighbors_hidden0')

  with tf.variable_scope('encode_neighbors1') as scope:
    h2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(h1, W1tiled), encoding_biases[1]),
                    name=scope.name + '/neighbors_hidden1')

  return h2




