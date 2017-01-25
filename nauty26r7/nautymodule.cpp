#include <Python.h>
#include <cstdint>
#include <stdio.h>

#include "nausparse.h"    /* which includes nauty.h */

#ifndef NDEBUG
#define ASSERT(condition, message) \
      do { \
                if (! (condition)) { \
                              std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                                        << " line " << __LINE__ << ": " << message << std::endl; \
                              std::terminate(); \
                          } \
            } while (false)
#else
#define ASSERT(condition, message) do { } while (false)
#endif

void print_adjlist(sparsegraph sg)
{
  for (size_t i = 0; i < sg.vlen; i++)
  {
    printf("%d: ", (int)i);
    for (size_t j = sg.v[i]; j < sg.v[i] + sg.d[i]; j++)
    {
      printf("%d, ", sg.e[j]);
    }
    printf("\n");
  }
}

static int* pylist2intarray(PyObject* item, int &len) {
    PyObject* seq = PySequence_Fast(item, "pylist2intarray: each PyObject row must be iterable.");
    len = PySequence_Fast_GET_SIZE(seq);
    int* arr = (int*) malloc(len * sizeof(int));
    if (!arr) {
      Py_DECREF(seq);
      len = -1;
      PyErr_NoMemory();
      return NULL;
    }
    for (int i = 0; i < len; i++) {
      PyObject *elemobj = PySequence_Fast_GET_ITEM(seq, i);
      if (!elemobj) {
        Py_DECREF(seq);
        free(arr);
        PyErr_SetString(PyExc_RuntimeError, "Failed to get item.");
        len = -1;
        return NULL;
      }

      PyObject *elem;
      elem = PyNumber_Int(elemobj);
      if (!elem) {
        Py_DECREF(seq);
        free(arr);
        PyErr_SetString(PyExc_TypeError, "All items must be integers");
        len = -1;
        return NULL;
      }

      arr[i] = (int) PyInt_AS_LONG(elem);
      Py_DECREF(elem);
      
    }
    return arr;

}

static sparsegraph adjlist2sparsegraph(int** adjlist, int* degrees, int num_nodes) {
  int n = num_nodes;
  sparsegraph sg;
  SG_INIT(sg);
  
  int num_edges = 0;
  for (int i = 0; i < n; i++) {
    num_edges += degrees[i];
  }

  SG_ALLOC(sg, n, num_edges, "malloc");

  sg.nv = n;              /* number of vertices */
  sg.nde = num_edges;     /* number of directed edges */

  memcpy(sg.d, degrees, sizeof(int) * n);

  sg.v[0] = 0;
  for (int i = 1; i < n; i++) {
    sg.v[i] = sg.v[i-1] + degrees[i-1];
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < degrees[i]; j++) {
      sg.e[sg.v[i] + j] = adjlist[i][j];
    }
  }

  return sg;
}

/* Expose sparse nauty canonical labelling */
static PyObject *nauty_sparsenauty(PyObject *self, PyObject *args)
{
  PyObject* seq;
  // Used for coloring in nauty
  PyObject* labseq = NULL;
  PyObject* ptnseq = NULL;

  int** adjlist;
  int* degrees;
  int seqlen;
  int i;

  /* get one argument as a sequence */
  if(!PyArg_ParseTuple(args, "O|(OO):sparsenauty", &seq, &labseq, &ptnseq))
    return 0;
  seq = PySequence_Fast(seq, "argument must be iterable");
  if(!seq)
    return 0;
  seqlen = PySequence_Fast_GET_SIZE(seq);

  /* prepare data as an array of doubles */
  adjlist = (int**) malloc(seqlen * sizeof(int*));

  /* handle oom */
  if(!adjlist) {
    Py_DECREF(seq);
    return PyErr_NoMemory(  );
  }
  degrees = (int*) malloc(seqlen * sizeof(int));
  if(!degrees) {
    Py_DECREF(seq);
    free(adjlist);
    return PyErr_NoMemory(  );
  }

  /* fill in rows */ 
  for(i=0; i < seqlen; i++) {
   
    int len;
    PyObject* rowobj = PySequence_Fast_GET_ITEM(seq, i);
    if (!rowobj) {
      Py_DECREF(seq);
      free(adjlist);
      free(degrees);
      PyErr_SetString(PyExc_RuntimeError, "Failed to get item.");
      return 0;
    }
    adjlist[i] = pylist2intarray(rowobj, len); 
    degrees[i] = len;
  }    

  Py_DECREF(seq);

  static DEFAULTOPTIONS_SPARSEGRAPH(options);
  //options.writeautoms = TRUE;
  options.getcanon = 1;
  statsblk stats;

  int n = seqlen;
  int m = SETWORDSNEEDED(n);
  nauty_check(WORDSIZE, m, n, NAUTYVERSIONID);

  DYNALLSTAT(int, orbits, orbits_sz);
  DYNALLOC1(int, orbits, orbits_sz, n, "malloc");

  DYNALLSTAT(int, lab, lab_sz);
  DYNALLSTAT(int, ptn, ptn_sz);
  if (labseq && ptnseq) {
    int lablen = 0;
    lab = pylist2intarray(labseq, lablen);
    ASSERT(seqlen == lab_sz, "Label sequence length does not match number of vertices in adjlist.");
    int ptnlen = 0;
    ptn = pylist2intarray(ptnseq, ptnlen);
    ASSERT(seqlen == ptn_sz, "Ptn sequence length does not match number of vertices in adjlist.");
    /* This option enables color constraints */
    options.defaultptn = false;
  } else {
    printf("no color\n");
    DYNALLOC1(int, lab, lab_sz, n, "malloc");
    DYNALLOC1(int, ptn, ptn_sz, n, "malloc");
  }

  sparsegraph g = adjlist2sparsegraph(adjlist, degrees, seqlen);
  sparsegraph canong;
  SG_INIT(canong);
  sparsenauty(&g, lab, ptn, orbits, &options, &stats, &canong);

  /* Automorphism sanity check only */
  /*
  printf("Automorphism group size = ");
  writegroupsize(stdout,stats.grpsize1,stats.grpsize2);
  printf("\n");
  */

  sortlists_sg(&canong);
  /*
  printf("original:\n");
  print_adjlist(g);
  printf("canon:\n");
  print_adjlist(canong);
  */

  // free arrays
  free(degrees);
  for (int i = 0; i < seqlen; i++) {
    free(adjlist[i]);
  }
  free(adjlist);

  PyObject* labellistobj = PyList_New(0);
  for (int i = 0; i < n; i++) {
    PyList_Append(labellistobj, Py_BuildValue("i", lab[i]));
  }

  return labellistobj;
}

static PyMethodDef nautyMethods[] = {
  {"canon_sparse", nauty_sparsenauty, METH_VARARGS, "Sum a sequence of numbers."}, {0} /* sentinel */
};

PyMODINIT_FUNC
initnauty(void)
{
      (void) Py_InitModule("nauty", nautyMethods);
}

