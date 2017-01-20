from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as nx

def read_adjlist_undir(filename):
  """ Read adjacency list from txt files.

  Args:
    filename: name of file containing adj list.

  Returns:
    Graph in networkx data structure.
  """
  G = nx.Graph()
  with open(filename, 'r') as gfile:
    for line in gfile:
      if (line[0] == '#'):
        continue
      else:
        edge = line.split()
        G.add_edge(edge[0], edge[1])
  return G

