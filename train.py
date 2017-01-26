
from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import tensorflow as tf
import time

import autoencoder as ae
import readgraph
import receptive_field as rf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_log_dir', '../log',
                           """Event logs and checkpoints during training.""")
tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('graph_dir', '../data/BlogCatalog-labeled/data',
                           """Directory containing graph data.""")
tf.app.flags.DEFINE_string('edge_filename', 'edges.csv',
                           """Name of file containing graph edges.""")
tf.app.flags.DEFINE_string('edge_delimiter', ',',
                           """Delimiter of edge file.""")
tf.app.flags.DEFINE_string('node_label_filename', 'group-edges.csv',
                           """Name of file containing node labels.""")
tf.app.flags.DEFINE_string('node_label_delimiter', ',',
                           """Delimiter of node label file.""")
tf.app.flags.DEFINE_integer('receptive_field_size', 150,
                           """Size of receptive field around each node.""")

NUM_EPOCHS = 20
BATCH_SIZE = 50
GPU_MEM_FRACTION = 0.6


def gen_feature(G, receptive_field):
  feature = []
  for i in range(len(receptive_field)):
    nodes = G.nodes()
    feature.append(nx.degree(G, nodes[receptive_field[i]]))
  if len(feature) < FLAGS.receptive_field_size:
    feature += [0] * (FLAGS.receptive_field_size - len(feature))
  return feature

def gen_node_feature(G, nodeid):
  candidates = rf.receptive_field_candidates(G, nodeid, FLAGS.receptive_field_size)
  candidates_flattened = [x for l in candidates for x in l]
  print('candidates: %s' % len(candidates_flattened))
  print('node: %s' % nodeid)
  degree = nx.degree(G, nodeid)
  print('degree: %s' % degree)
  localgraph = G.subgraph(candidates_flattened)
  ranking = rf.rank_candidates(localgraph, candidates) 
  return gen_feature(localgraph, ranking[:FLAGS.receptive_field_size])

def train(G):
  global_step = tf.Variable(0, trainable=False)
  
  n = nx.number_of_nodes(G)
  nodes = G.nodes()

  X = tf.placeholder('float', [None, FLAGS.receptive_field_size], 'input')
  h, weights = ae.encode(X)
  recon_x = ae.decode(h, X.get_shape()[1], weights)
  loss = ae.loss(recon_x, X)
  train_op = ae.train(loss, global_step)
  
  init = tf.global_variables_initializer()
  saver = tf.train.Saver(tf.global_variables())
  summary_op = tf.summary.merge_all()

  # Memory usage options
  config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
  config.gpu_options.allow_growth = True
  config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION

  # Start running operations on the Graph.
  sess = tf.Session(config=config)
  sess.run(init)

  summary_writer = tf.summary.FileWriter(FLAGS.train_log_dir, sess.graph)

  for epoch in range(NUM_EPOCHS):
    randperm = np.random.permutation(n)
    for i in range(n // BATCH_SIZE):
      batch_x = []
      for j in range(BATCH_SIZE):
        feature = gen_node_feature(G, nodes[randperm[i * BATCH_SIZE + j]])
        print('feature: %s' % feature)
        batch_x.append(feature)
      
      start_time = time.time()
      #_, loss_value = sess.run([train_op, loss], feed_dict={X: batch_x})
      duration = time.time() - start_time
      
      #assert not np.isnan(loss_value), 'Model diverged with loss = NaN'


def main(argv=None):
  if tf.gfile.Exists(FLAGS.train_log_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_log_dir)
    tf.gfile.MakeDirs(FLAGS.train_log_dir)

  G = readgraph.read_adjlist_undir(os.path.join(FLAGS.graph_dir, FLAGS.edge_filename), ',')
  G = readgraph.read_node_attribute(G, os.path.join(FLAGS.graph_dir, FLAGS.node_label_filename), 'group', ',')
  train(G)

if __name__ == '__main__':
  tf.app.run()

