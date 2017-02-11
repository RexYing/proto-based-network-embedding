from __future__ import division, print_function, absolute_import
from datetime import datetime
import networkx as nx
import numpy as np
import os
import tensorflow as tf
import time

import pandas as pd
from redditnetwork import constants
from redditnetwork.network_extractor import extract_week_network


def user_graph(G):
  """ Extract user-user induced graph from G.
  """
  
  userG = networkx.graph()
  return userG

def train(G):
  print(politics_net.graph)


def main(argv=None):
  if tf.gfile.Exists(FLAGS.train_log_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_log_dir)
    tf.gfile.MakeDirs(FLAGS.train_log_dir)

  G = extract_week_network("politics", 2014, 1)
  train(G)

if __name__ == '__main__':
  tf.app.run()

