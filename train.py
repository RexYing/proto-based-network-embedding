
from __future__ import division, print_function, absolute_import
from datetime import datetime
import networkx as nx
import numpy as np
import os
import tensorflow as tf
import time

from autoencoder import Autoencoder
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
tf.app.flags.DEFINE_string('node_filename', 'nodes.csv',
                           """Name of file containing graph edges.""")
tf.app.flags.DEFINE_string('edge_filename', 'edges.csv',
                           """Name of file containing graph edges.""")
tf.app.flags.DEFINE_string('edge_delimiter', ',',
                           """Delimiter of edge file.""")
tf.app.flags.DEFINE_string('node_label_filename', 'group-edges.csv',
                           """Name of file containing node labels.""")
tf.app.flags.DEFINE_string('node_label_delimiter', ',',
                           """Delimiter of node label file.""")
tf.app.flags.DEFINE_integer('receptive_field_size', 5,
                           """Size of receptive field around each node.""")
tf.app.flags.DEFINE_string('receptive_field_path', 'rf.txt',
                           """Saved features.""")
tf.app.flags.DEFINE_boolean('continue_receptive_field', False, """Continue feature extraction.""")

# training params
tf.app.flags.DEFINE_integer('batch_size', 30, """Batch size.""")
tf.app.flags.DEFINE_float('weight_decay', 0.0000, """Weight decay for regularization.""")
tf.app.flags.DEFINE_float('learning_rate', 0.001, """Initial learning rate.""")

NUM_EPOCHS = 40
GPU_MEM_FRACTION = 0.5


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
    if receptive_field[i] not in G:
      print('Node not in graph: %s' % receptive_field[i])
    feature.append(G.degree(receptive_field[i]))
  if len(feature) < FLAGS.receptive_field_size:
    feature += [0] * (FLAGS.receptive_field_size - len(feature))
  return feature

def gen_features(G):
  """ Generate feature for all nodes in G according to their receptive fields.
  """
  for node in G.nodes():
    G.node[node]['feature'] = gen_feature(G, G.node[node]['rf'])

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


def create_batch(G, randperm, batch_idx):
  nodes = G.nodes()
  batch_x = []
  batch_neighbors = []
  batch_degrees = []
  for j in range(FLAGS.batch_size):
    nodeid = nodes[randperm[batch_idx * FLAGS.batch_size + j]]
    curr_node = G.node[nodeid]
    batch_degrees.append(G.degree(nodeid))
    feature = curr_node['feature']
    batch_x.append(feature)
    neighbor_feature_list = [G.node[neighbor]['feature']  for neighbor in G.neighbors(nodeid)]
    batch_neighbors.append(neighbor_feature_list)

  # find the maximum degree among the batch of nodes and pad neighbor_feature_list to the max
  max_deg = np.max(batch_degrees)
  for i in range(len(batch_neighbors)):
    batch_neighbors[i] = np.pad(batch_neighbors[i], 
                                ((0, max_deg - len(batch_neighbors[i])), (0, 0)),
                                'constant')
  #print(np.array(batch_neighbors).shape)
  #print(np.array(batch_x).shape)
  return batch_x, batch_neighbors


def train(G):
  
  n = nx.number_of_nodes(G)
  nodes = G.nodes()

  X = tf.placeholder('float', [None, FLAGS.receptive_field_size], 'input')
  wd = tf.placeholder('float', [], 'weight_decay')
  dropout = tf.placeholder('float', [], 'dropout')

  layer_sizes = [5, 5, 5]
  placeholders = {'features': X, 'weight_decay': wd, 'dropout': dropout}
  model = Autoencoder(placeholders, layer_sizes, tied_weights=True)
  
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
        receptive_field = rf.strip().split(',')
        G.node[nodeid]['rf'] = receptive_field[:FLAGS.receptive_field_size]

  if (not tf.gfile.Exists(FLAGS.receptive_field_path)) or FLAGS.continue_receptive_field:
    print('Generate receptive fields')
    f = open(FLAGS.receptive_field_path, 'a+')
    for nodeid in G.nodes():
      if not 'rf' in G.node[nodeid]:
        receptive_field = gen_receptive_field(G, nodeid)
        G.node[nodeid]['rf'] = receptive_field

        line = str(nodeid) + ' '
        line += str(receptive_field[0])
        for i in receptive_field[1:]:
          line += ',' + str(i)
        line += '\n'
        print(line)
        f.write(line)

  gen_features(G)
  nodes = G.nodes()
  step = 0


  for epoch in range(NUM_EPOCHS):
    print('Epoch: %d' % epoch)
    randperm = np.random.permutation(n)
    for batch_idx in range(n // FLAGS.batch_size):

      batch_x, batch_neighbors = create_batch(G, randperm, batch_idx)

      feeddict = {X: batch_x, wd: FLAGS.weight_decay}

      
      start_time = time.time()
      _, loss_value = sess.run([model.opt_op, model.loss], feed_dict=feeddict)
      #_, loss_value = sess.run([train_op, loss], feed_dict=feeddict)
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 1 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (datetime.now(), step, loss_value, examples_per_sec,
                                 sec_per_batch))

      if step % 10 == 0:
        summary_str = sess.run(summary_op, feed_dict=feeddict)
        summary_writer.add_summary(summary_str, step)

      step += 1


def main(argv=None):
  if tf.gfile.Exists(FLAGS.train_log_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_log_dir)
    tf.gfile.MakeDirs(FLAGS.train_log_dir)

  G = readgraph.read_adjlist_undir(os.path.join(FLAGS.graph_dir, FLAGS.edge_filename), ',')
  readgraph.add_nodes(G, os.path.join(FLAGS.graph_dir, FLAGS.node_filename))
  G = readgraph.read_node_attribute(G, os.path.join(FLAGS.graph_dir, FLAGS.node_label_filename), 'group', ',')
  train(G)

if __name__ == '__main__':
  tf.app.run()

