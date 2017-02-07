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


def wl_color_refinement(G, labels, sparse=True):
  """ Color refinement using WL algorithm.

  Args:
    G: input graph.
    labels: List of labels for nodes in G, ordered according to G.nodes().
    sparse: True if G is sparse

  Returns:
    labels: Refined labels
  """
  num_nodes = len(labels)

  prime_upper = primes_arguments_required[int(np.ceil(np.log2(num_nodes))) + 1]
  logplist = np.log2(primes(prime_upper))

  # Use adj matrix multiplication if the graph is not sparse
  if not sparse:
    adjmat = nx.adjacency_matrix(G)
    signatures = np.round(labels + adjmat.dot([logplist[i] for i in labels]), decimals=5)
  else:
    nodes = G.nodes()
    # perfect hash
    nodeid2hash = {nodes[i]: logplist[labels[i]] for i in range(num_nodes)}
    signatures = []
    for curr_node in nodes:
      hash_val = 0
      for neighbor in G.neighbors(curr_node):
        hash_val += nodeid2hash[neighbor]
      signatures.append(hash_val)
  
  _, newlabels = np.unique(signatures, return_inverse=True)

  return newlabels

def wl(G, labels=[], steps=0):
  """ Label local neighborhood with WL (Weisfeiler-Lehman) algorithm.
    The distance constraint applies: if d(u, v) < d(w, v), l(u) < l(w)

  Args:
    G: input local neighborhood.

  Returns:
    labels: list of (refined) labels
  """
  if not labels:
    labels = [1] * len(G.nodes())
  prev_labels = []
  num_iter = 0

  while np.any(prev_labels != labels):
    prev_labels = labels
    labels = wl_color_refinement(G, labels)
    num_iter += 1
    if (steps > 0 and num_iter == steps):
      break

  return labels

