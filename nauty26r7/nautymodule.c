#include <Python.h>

#include "nausparse.h"    /* which includes nauty.h */

/* a preexisting C-level function you want to expose -- e.g: */
static double total(double* data, int len)
{
  double total = 0.0;
  int i;
  for(i=0; i<len; ++i)
    total += data[i];
  return total;
}

/* here is how you expose it to Python code: */
static PyObject *nauty_sparsenauty(PyObject *self, PyObject *args)
{
  PyObject* seq;
  double *dbar;
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
  dbar = malloc(seqlen*sizeof(double));
  if(!dbar) {
    Py_DECREF(seq);
    return PyErr_NoMemory(  );
  }
  for(i=0; i < seqlen; i++) {
    PyObject *fitem;
    PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
    if(!item) {
      Py_DECREF(seq);
      free(dbar);
      return 0;
    }
    fitem = PyNumber_Float(item);
    if(!fitem) {
      Py_DECREF(seq);
      free(dbar);
      PyErr_SetString(PyExc_TypeError, "all items must be numbers");
      return 0;
    }
    dbar[i] = PyFloat_AS_DOUBLE(fitem);
    Py_DECREF(fitem);
  }    

  /* clean up, compute, and return result */
  Py_DECREF(seq);
  result = total(dbar, seqlen);
  free(dbar);
  return Py_BuildValue("d", result);
}

static PyMethodDef nautyMethods[] = {
  {"canon_sparse", nauty_sparsenauty, METH_VARARGS, "Sum a sequence of numbers."}, {0} /* sentinel */
};

PyMODINIT_FUNC
initnauty(void)
{
      (void) Py_InitModule("nauty", nautyMethods);
}

