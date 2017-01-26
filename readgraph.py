from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as nx

def read_node_attribute(G, filename, attr_name, delimiter):
  """ Read label from txt/csv files.

  Args:
    filename: name of file containing node and label.
    attr_name: name of attribute.
    delimiter: delimiter that separates node and label.

  Returns:
    Graph with labels
  """
  with open(filename, 'r') as attrfile:
    for line in attrfile:
      [id, attr] = line.strip().split(',')
      G.node[id][attr_name] = attr
  return G


def read_adjlist_undir(filename, delimiter):
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
        edge = line.strip().split(delimiter)
        G.add_edge(edge[0], edge[1])
  return G

