from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import nauty

if __name__ == '__main__':
    adjlist = [[2,1],[0,3],[0,4],[1,4],[3,2]]
    lab = [1, 4, 2, 0, 3]
    ptn = [1, 0, 1, 1, 0]

    print("regular graph with 5 vertices and 10 edges: without labels")
    canon_lab = nauty.canon_sparse(adjlist)
    print("canon label: %s " % canon_lab)

    print("regular graph with 5 vertices and 10 edges: with labels")
    canon_lab = nauty.canon_sparse(adjlist, (lab, ptn))
    print("canon label: %s " % canon_lab)

