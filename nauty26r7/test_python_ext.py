import nauty

if __name__ == '__main__':
    adjlist = [[2,1],[0,3],[0,4],[1,4],[3,2]]
    nauty.canon_sparse(adjlist)

