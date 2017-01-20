from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import readgraph
import local_label as label

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
  G.node[node]['depth'] = 0
  while (n_candidates < k):
    next_depth = []
    for i in candidates[depth]:
      neighbors = G.neighbors(i)
      neighbors = [x for x in neighbors if (not 'depth' in G.node[x])]
      for x in neighbors:
        G.node[x]['depth'] = depth + 1
      next_depth.extend(neighbors)
      n_candidates += len(neighbors)
    depth += 1
    candidates.append(next_depth)
  return candidates 


def rank_candidates(G):
  labels = label.wl(G)
  return G
    

if __name__ == '__main__':
  G = readgraph.read_adjlist_undir('data/ca-GrQc.txt')

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

  rank_candidates(localgraph) 
