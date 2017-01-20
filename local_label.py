from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import networkx as nx
import readgraph

# (2^i)-th prime
primes_arguments_required = [2, 3, 7, 19, 53, 131, 311, 719, 1619,
                             3671, 8161, 17863, 38873, 84017, 180503, 386093,
                             821641, 1742537, 3681131, 7754077, 16290047];

def primes(n):
  """ Generate a list of primes less than or equal to n.
  """
  out = list()
  sieve = [True] * (n+1)
  for p in range(2, n+1):
    if (sieve[p]):
      out.append(p)
      for i in range(p, n+1, p):
        sieve[i] = False
  return out


def wl_color_refinement(G, labels):
  """ Color refinement using WL algorithm.

  Args:
    G: input graph.
    labels: List of labels for nodes in G, ordered according to G.nodes().
  """
  num_alph = len(labels)

  prime_upper = primes_arguments_required[np.ceil(np.log2(num_alph))]
  logplist = np.log2(primes(prime_upper))

  adjmat = nx.adjacency_matrix(G)
  signatures = np.round(labels + adjmat.dot(logplist), decimals=10)
  _, newlabels = np.unique(signatures, return_inverse=True)
  print(signatures)
  print(newlabels)

def wl(G, labels=[]):
  """ Label local neighborhood with WL (Weisfeiler-Lehman) algorithm.
    The distance constraint applies: if d(u, v) < d(w, v), l(u) < l(w)

  Args:
    G: input local neighborhood.

  Returns:
    G with labels
  """
  if not labels:
    labels = [1] * len(G.nodes())


def receptive_field_candidates(G, node, k):
  """ Return candidates of CNN receptive field using bfs.

  Args:
    node: source node whose neighborhood needs to be constructed.
    k: receptive field size.

  Returns:
    candidates: a list of node IDs that are candidates for receptive field
  """
  candidates = [node]
  idx = 0;
  while (len(candidates) < k):
    l = len(candidates)
    for i in range(idx, l):
      candidates.extend(G.neighbors(candidates[i]))
  return candidates 
    

if __name__ == '__main__':
  G = readgraph.read_adjlist_undir('../data/ca-GrQc.txt')

  node = G.nodes()[0]
  print('node : %s' % node)
  candidates = receptive_field_candidates(G, node, 10)
  print('candidates : %s' % candidates) 
  nx.draw(G.subgraph(candidates))

