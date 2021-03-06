from __future__ import division, print_function, absolute_import
from datetime import datetime
import math
import networkx as nx
import numpy as np
import os
import pandas as pd
import sklearn as sk
import tensorflow as tf
import time

from models import GCN_multipartite
from redditnetwork import constants
from redditnetwork.network_extractor import extract_week_network
import utils

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_log_dir', '../log',
                           """Event logs and checkpoints during training.""")
tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

# training params
tf.app.flags.DEFINE_integer('epochs', 200, """Number of epochs to train.""")
tf.app.flags.DEFINE_integer('batch_size', 30, """Batch size.""")
tf.app.flags.DEFINE_float('weight_decay', 0.0000, """Weight decay for regularization.""")
tf.app.flags.DEFINE_float('learning_rate', 0.001, """Initial learning rate.""")
tf.app.flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')

GPU_MEM_FRACTION = 0.5

POST_TYPE = 'post'
COMMENT_TYPE = 'comment'
USER_TYPE = 'user'


def extract_posts(G):
  return [nodeid for nodeid in G.nodes() if G.node[nodeid]['type'] == POST_TYPE]

def get_author(G, nodeid):
  """ Get the user for a post/comment in graph
  """
  for predid in G.predecessors(nodeid):
    if G.node[predid]['type'] == USER_TYPE:
      return predid

def user_graph(G):
  """ Extract user-user induced graph from G.
    Users are connected if they post/comment on the same thread.
  """
  userG = nx.Graph()
  postids = extract_posts(G)
  for postid in postids:
    # bfs on comments
    # assume the thread is acyclic and thus there is no visited check 
    comment_queue = [(postid, 0)]
    while len(comment_queue) > 0:
      cmtid, depth = comment_queue.pop(0)
      author = get_author(G, cmtid)
      for succid in G.successors(cmtid):
        if not G.node[succid]['type'] == COMMENT_TYPE:
          print('Successor of post/comment %s is of type %s' % (cmtid, G.node[succid]['type']))
          break
        comment_queue.append((succid, depth + 1))
        succ_author = get_author(G, succid)
        if G.has_edge(author, succ_author):
          userG[author][succ_author]['count'] += 1
        else:
          userG.add_edge(author, succ_author)
          userG[author][succ_author]['count'] = 1
  return userG

def post_raw_feature(G, postid):
  feature = [G.node[postid]['score'], G.node[postid]['length']] 
  feature.extend(G.node[postid]['word_vecs'])
  return feature

def comment_raw_feature(G, commentid):
  feature = [G.node[commentid]['score'], G.node[commentid]['length'], G.node[commentid]['post_time_offset']]
  feature.extend(G.node[commentid]['word_vecs'])
  return feature

def extract_user_features_simple(G, userid, max_deg, neighbor_dict=None):
  """ Extract user graph and features for each user.
  Extracted features are based on neighboring posts/comments

  Args:
    G: networkx graph with nodes representing comments, posts and users.
    max_deg: the first dimension of the feature. If the number of comments and posts that the user
        makes is less than max_degree, pad the remaining rows with zeros.
    neighbor_dict: dictionary mapping post/comment ids to a tuple of index and 

  Returns:
    features: features for each user.
  """
  post_feat_len = G.graph['post_feats']['score'] + G.graph['post_feats']['length'] + \
      G.graph['post_feats']['word_vec']
  post_feat = []
  comment_feat_len = G.graph['comment_feats']['score'] + G.graph['comment_feats']['length'] + \
      G.graph['comment_feats']['post_time_offset'] + G.graph['comment_feats']['word_vec']
  comment_feat = []
  feature_size = post_feat_len + comment_feat_len

  feature_idx = []
  for neighbor in G.neighbors(userid):
    if neighbor_dict is not None and neighbor in neighbor_dict:
      feature_idx.append(neighbor_dict[neighbor][0])
    elif G.node[neighbor]['type'] == POST_TYPE: 
      f = post_raw_feature(G, neighbor)
      post_feat.append(f)
      comment_feat.append([0.0] * comment_feat_len)
      if neighbor_dict is not None:
        feature_idx.append(len(neighbor_dict))
        neighbor_dict[neighbor] = (len(neighbor_dict), np.concatenate((f, [0.0] * comment_feat_len)))
    elif G.node[neighbor]['type'] == COMMENT_TYPE:
      f = comment_raw_feature(G, neighbor)
      post_feat.append([0.0] * post_feat_len)
      comment_feat.append(f)
      if neighbor_dict is not None:
        feature_idx.append(len(neighbor_dict))
        neighbor_dict[neighbor] = (len(neighbor_dict), np.concatenate(([0.0] * post_feat_len, f)))
    else:
      print('Unknown type %s' % G.node[neighbor]['type'])
  feature = np.concatenate((post_feat, comment_feat), 1)
  feature = np.pad(feature, ((0, max_deg - feature.shape[0]), (0, 0)), 'constant')
  feature_idx = np.pad(feature_idx, (0, max_deg - len(feature_idx)), 'constant')
  #print(feature_idx)
  if neighbor_dict is None:
    return feature_size, feature
  else:
    return feature_size, feature_idx

# Define model evaluation function
def evaluate(sess, model, features, support, labels, mask, placeholders):
  t_test = time.time()
  feed_dict_val = utils.construct_feed_dict(features, support, labels, mask, placeholders,
      sparse_inputs=False)
  outs_val = sess.run([model.loss, model.accuracy, model.y_pred], feed_dict=feed_dict_val)
  return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)

def train(G):
  print(G.graph)
  print('Extracting user graph...')
  userG = user_graph(G)

  # labels
  print('Obtaining labels...')
  future = pd.read_csv(constants.DATA_HOME + "user_scores/{}_2014_wf{:02d}.csv".format("politics", 1))
  future = future.set_index("user")
  future['label'] = np.sign(future["sum"] - np.percentile(future["sum"], 90) - 1e-10)
  labels = []
  for userid in userG.nodes():
    if userid in future.index:
      labels.append(future.loc[userid]['label'])
    else:
      #print('%s does not have label.' % userid)
      userG.remove_node(userid)
  labels = np.array(labels, dtype=np.int)
  # convert to 0/1 labels
  labels = [0 if l < 0 else 1 for l in labels]

  print('Extracting user features...')
  max_deg = max(G.degree(userG.nodes()).values())
  print('Max user degree: ', max_deg)
  features = []
  neighbor_dict = {}
  feature_size = 0
  for node in userG.nodes():
    #feature_size, feature_idx = extract_user_features_simple(G, node, max_deg, neighbor_dict)
    feature_size, feature_idx = extract_user_features_simple(G, node, max_deg)
    features.append(feature_idx)

  # a list of all features corresponding to posts/comments
  #neighbor_features = np.zeros((len(neighbor_dict), feature_size))
  #for idx, vec in neighbor_dict.values():
  #  neighbor_features[idx] = vec
  features = np.stack(features, axis=0)
  #features = np.array(features)
  print('Feature dimensions: ', features.shape)

  # data split
  n = userG.number_of_nodes()
  n1 = int(math.ceil(n * 0.7))
  n2 = int(math.ceil(n * 0.8))
  train_mask = np.array([1 if i < n1 else 0 for i in range(n)])
  val_mask = np.array([1 if n1 <= i < n2 else 0 for i in range(n)])
  test_mask = np.array([1 if n2 <= i else 0 for i in range(n)])

  train_labels = np.zeros((n, 2))
  train_labels[np.arange(n1), labels[:n1]] = 1
  val_labels = np.zeros((n, 2))
  val_labels[np.arange(n1, n2), labels[n1:n2]] = 1
  test_labels = np.zeros((n, 2))
  test_labels[np.arange(n2, n), labels[n2:]] = 1

  adj = nx.adjacency_matrix(userG)

  # Define placeholders
  placeholders = {
      'support': [tf.sparse_placeholder(tf.float32)],
      'features': tf.placeholder(tf.float32, shape=(None, features.shape[1], features.shape[2])),
      #'features': tf.placeholder(tf.float32, shape=(None, features.shape[1])),
      #'neighbor_features': tf.placeholder(tf.float32, shape=neighbor_features.shape),
      'labels': tf.placeholder(tf.float32, shape=(None, train_labels.shape[1])),
      'labels_mask': tf.placeholder(tf.int32),
      'dropout': tf.placeholder_with_default(0., shape=()),
  }

  # neural network model
  layer_sizes = [features.shape[2], 10, 1]
  model = GCN_multipartite(placeholders, layer_sizes, logging=True)

  # Memory usage options
  config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
  config.gpu_options.allow_growth = True
  config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION

  init = tf.global_variables_initializer()
  saver = tf.train.Saver(tf.global_variables())
  summary_op = tf.summary.merge_all()

  # Start running operations on the Graph.
  sess = tf.Session(config=config)
  sess.run(init)
  summary_writer = tf.summary.FileWriter(FLAGS.train_log_dir, sess.graph)
  
  cost_val = []

  # Train model
  print('Training...')
  for epoch in range(FLAGS.epochs):

    support = [utils.preprocess_adj(adj)]
    # Construct feed dictionary
    feed_dict = utils.construct_feed_dict(features, support, train_labels, train_mask,
        placeholders, sparse_inputs=False)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    start_time = time.time()
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
    duration = time.time() - start_time
    print(duration)

    # Validation
    cost, acc, y_pred, duration_val = evaluate(sess, model, features, support, val_labels, val_mask, placeholders)
    cost_val.append(cost)

    y_true = np.argmax(val_labels,1)
    y_pred = y_pred[n1:n2]
    y_true = y_true[n1:n2]
    precision = sk.metrics.precision_score(y_true, y_pred)
    recall = sk.metrics.recall_score(y_true, y_pred)
    f1 = sk.metrics.f1_score(y_true, y_pred)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), 
          "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), 
          "train_time=", "{:.5f}".format(duration),
          "val_acc=", "{:.5f}".format(acc), 
          'val_f1=', '{:.5f}'.format(f1),
          "val_time=", "{:.5f}".format(duration_val))


    summary_str = sess.run(summary_op, feed_dict=feed_dict)
    summary_writer.add_summary(summary_str, epoch)


def main(argv=None):
  if tf.gfile.Exists(FLAGS.train_log_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_log_dir)
    tf.gfile.MakeDirs(FLAGS.train_log_dir)

  print('Extracting reddit network...')
  G = extract_week_network("politics", 2014, 1)
  train(G)

if __name__ == '__main__':
  tf.app.run()

