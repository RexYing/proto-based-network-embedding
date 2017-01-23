#include <Python.h>
#include <cstdint>
#include <stdio.h>

#include "nausparse.h"    /* which includes nauty.h */


static int* pylist2intarray(PyObject* item, int &len) {
    PyObject* seq = PySequence_Fast(item, "each adjacency row must be iterable");
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
  SG_ALLOC(sg, n, 0, "malloc");

  sg.nv = n;      /* number of vertices */
  sg.nde = 0;     /* number of directed edges */

  memcpy(sg.d, degrees, sizeof(int) * n);

}

/* Expose sparse nauty canonical labelling */
static PyObject *nauty_sparsenauty(PyObject *self, PyObject *args)
{
  PyObject* seq;
  int** adjlist;
  int* degrees;
  double result;
  int seqlen;
  int i;

  /* get one argument as a sequence */
  if(!PyArg_ParseTuple(args, "O", &seq))
    return 0;
  seq = PySequence_Fast(seq, "argument must be iterable");
  if(!seq)
    return 0;

  /* prepare data as an array of doubles */
  seqlen = PySequence_Fast_GET_SIZE(seq);
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
      free(dbar);
      PyErr_SetString(PyExc_RuntimeError, "Failed to get item.");
      return 0;
    }
    adjlist[i] = pylist2intarray(rowobj, len); 
    degrees[i] = len;
    printf("row size: %d  ", len);
  }    

  Py_DECREF(seq);

  sparsegraph g = adjlist2sparsegraph(adjlist, degrees, seqlen);

  // free arrays
  free(adjlist);
  free(degrees);
  for (int i = 0; i < seqlen; i++) {
    free(adjlist[i]);
  }

  return Py_BuildValue("i", seqlen);
}

static PyMethodDef nautyMethods[] = {
  {"canon_sparse", nauty_sparsenauty, METH_VARARGS, "Sum a sequence of numbers."}, {0} /* sentinel */
};

PyMODINIT_FUNC
initnauty(void)
{
      (void) Py_InitModule("nauty", nautyMethods);
}

