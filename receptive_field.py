from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import readgraph
import local_label as label
import nauty

def receptive_field_candidates(G, node, k):
  """ Return candidates of CNN receptive field using bfs.

  Args:
    node: source node whose neighborhood needs to be constructed.
    k: receptive field size.

  Returns:
    candidates: 2D list containing candidates for receptive field.
      The i-th list in candidates contains nodes that are i hops away from the source.
  """
  n_candidates = 1
  candidates = [[node]]
  depth = 0;
  depths = {}
  depths[node] = 0
  while (n_candidates < k):
    next_depth = []
    for i in candidates[depth]:
      neighbors = G.neighbors(i)
      neighbors = [x for x in neighbors if (not x in depths)]
      for x in neighbors:
        depths[x] = depth
      next_depth.extend(neighbors)
      n_candidates += len(neighbors)
    depth += 1
    if not next_depth:
      break
    candidates.append(next_depth)
  return candidates 


def rank_candidates(G, candidates):
  labels = label.wl(G)
  print('labels: %s' % labels)

  id2local = {val: key for (key, val) in list(enumerate(G.nodes()))}

  # Build labels and ptns for canonicalization
  lab = []
  ptn = []

  for nodelist in candidates:
    curr_local_ids = [id2local[id] for id in nodelist]
    curr_lvl_labels = [labels[id2local[id]] for id in nodelist]

    _, newlabels = np.unique(curr_lvl_labels, return_inverse=True)
    # nodes with the same label have the same color
    lab2nodes = [[] for _ in range(np.max(newlabels) + 1)]
    for i in range(len(newlabels)):
      lab2nodes[newlabels[i]].append(curr_local_ids[i])
    for i in range(len(lab2nodes)):
      lab.extend(lab2nodes[i])
      ptn.extend([1 for _ in range(len(lab2nodes[i]) - 1)])
      ptn.append(0)

  adjlist = []
  for node in G.nodes():
    adjlist.append([id2local[i] for i in G.neighbors(node)])
  print('Running canonicalization')
  canon_lab = nauty.canon_sparse(adjlist, (lab, ptn))
  return canon_lab
    

if __name__ == '__main__':
  #G = readgraph.read_adjlist_undir('data/ca-GrQc.txt')
  G = readgraph.read_adjlist_undir('../data/BlogCatalog-labeled/data/edges.csv', ',')
  G = readgraph.read_node_attribute(G, '../data/BlogCatalog-labeled/data/group-edges.csv', 'group', ',')

  node = G.nodes()[0]
  print('neighborhood of node : %s' % node)
  candidates = receptive_field_candidates(G, node, 10)
  candidates_flattened = [x for l in candidates for x in l]

  localgraph = G.subgraph(candidates_flattened)
  pos=nx.spring_layout(localgraph)
  nx.draw_networkx_nodes(localgraph, pos, nodelist=[candidates[0][0]], node_color='b', node_size=500)
  nx.draw_networkx_nodes(localgraph, pos, nodelist=candidates_flattened[1:], node_color='r', node_size=400)
  nx.draw_networkx_edges(localgraph, pos, width=1.0, alpha=0.5)
  plt.show()

  canon_lab = rank_candidates(localgraph, candidates) 

