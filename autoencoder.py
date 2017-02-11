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




