
from __future__ import division, print_function, absolute_import

#import matplotlib.pyplot as plt
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
tf.app.flags.DEFINE_integer('receptive_field_size', 200,
                           """Size of receptive field around each node.""")
tf.app.flags.DEFINE_string('receptive_field_path', 'rf.txt',
                           """Saved features.""")
tf.app.flags.DEFINE_boolean('continue_receptive_field', False, """Continue feature extraction.""")

NUM_EPOCHS = 20
BATCH_SIZE = 30
GPU_MEM_FRACTION = 0.4
CONTRASTIVE_REG = 0.001


def gen_feature(G, receptive_field):
  """ Generate feature for nodes in receptive field

  Args:
    G: Graph that contains nodes in receptive field, used to generate structure-related features.
    receptive_field: List containing IDs of nodes in receptive field.

  Returns:
    A feature vector containing concatenation of features for nodes in receptive field
  """
  feature = []
  for i in range(len(receptive_field)):
    feature.append(nx.degree(G, receptive_field[i]))
  if len(feature) < FLAGS.receptive_field_size:
    feature += [0] * (FLAGS.receptive_field_size - len(feature))
  return feature

def gen_receptive_field(G, nodeid):
  candidates = rf.receptive_field_candidates(G, nodeid, FLAGS.receptive_field_size)
  candidates_flattened = [x for l in candidates for x in l]
  #print('candidates: %s' % len(candidates_flattened))
  print('node: %s' % nodeid)
  localgraph = G.subgraph(candidates_flattened)
  ranking = rf.rank_candidates(localgraph, candidates) 

  nodes = localgraph.nodes()
  receptive_field = [nodes[ranking[i]] for i in range(0, FLAGS.receptive_field_size)]
  return receptive_field

def gen_node_feature(G, nodeid):
  receptive_field = gen_receptive_field(G, nodeid)
  return gen_feature(G, receptive_field)

def train(G):
  global_step = tf.Variable(0, trainable=False)
  
  n = nx.number_of_nodes(G)
  nodes = G.nodes()

  reg_multiplier = tf.placeholder('float', [1,], 'reg')
  X = tf.placeholder('float', [None, FLAGS.receptive_field_size], 'input')
  h, weights, biases, gradf = ae.encode(X)
  recon_x = ae.decode(h, X.get_shape()[1], weights)
  # gradient wrt input for constrastive regularization
  loss = ae.loss(recon_x, X, gradf, reg_multiplier)
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

  if tf.gfile.Exists(FLAGS.receptive_field_path):
    with open(FLAGS.receptive_field_path, 'r') as rffile:
      for line in rffile:
        [nodeid, rf] = line.strip().split()
        G.node[nodeid]['feature'] = gen_feature(rf.strip().split(','))

  if (not tf.gfile.Exists(FLAGS.receptive_field_path)) or FLAGS.continue_receptive_field:
    print('Generate receptive fields')
    f = open(FLAGS.receptive_field_path, 'a+')
    for nodeid in G.nodes():
      #feature = gen_node_feature(G, nodeid)
      #G.node[nodeid]['feature'] = feature
      line = str(nodeid) + ' '
      receptive_field = gen_receptive_field(G, nodeid)
      line += str(receptive_field[0])
      for i in receptive_field[1:]:
        line += ',' + str(i)
      line += '\n'
      print(line)
      f.write(line)


  for epoch in range(NUM_EPOCHS):
    randperm = np.random.permutation(n)
    for i in range(n // BATCH_SIZE):
      batch_x = []
      batch_neighbors = []
      batch_degrees = []
      for j in range(BATCH_SIZE):
        curr_node = G.node[randperm[i * BATCH_SIZE + j]]
        batch_degrees.append(G.degree(curr_node))
        feature = curr_node['feature']
        batch_x.append(feature)
        neighbor_feature_list = [neighbor['feature']  for neighbor in G.neighbors(curr_node)]
        batch_neighbors.append(neighbor_feature_list)

      # find the maximum degree among the batch of nodes and pad neighbor_feature_list to the max
      max_deg = np.max(batch_degrees)
      print('max deg: %d' % max_deg)
      for i in range(len(batch_neighbors)):
        batch_neighbors[i] = np.pad(batch_neighbors[i], 
                                    ((0, max_deg - len(batch_neighbors[i])), (0, 0)),
                                    'constant')
      print(batch_neighbors[0].shape)
      
      print(batch_neighbors.shape)
      print('batch_x shape: %s' % batch_x.shape)
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss], feed_dict={X: batch_x, reg_multiplier: CONTRASTIVE_REG})
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 1 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (datetime.now(), step, loss_value, examples_per_sec,
                                 sec_per_batch))


def main(argv=None):
  if tf.gfile.Exists(FLAGS.train_log_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_log_dir)
    tf.gfile.MakeDirs(FLAGS.train_log_dir)

  G = readgraph.read_adjlist_undir(os.path.join(FLAGS.graph_dir, FLAGS.edge_filename), ',')
  G = readgraph.read_node_attribute(G, os.path.join(FLAGS.graph_dir, FLAGS.node_label_filename), 'group', ',')
  train(G)

if __name__ == '__main__':
  tf.app.run()

