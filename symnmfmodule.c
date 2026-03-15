#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h" 

/* Helper function: Convert Python List of Lists to C double** matrix */
double** python_to_c_matrix(PyObject *list_obj, int rows, int cols) {
    int i, j;
    double **matrix = allocate_matrix(rows, cols);
    for (i = 0; i < rows; i++) {
        PyObject *row_list = PyList_GetItem(list_obj, i);
        for (j = 0; j < cols; j++) {
            matrix[i][j] = PyFloat_AsDouble(PyList_GetItem(row_list, j));
        }
    }
    return matrix;
}

/* Helper function: Convert C double** matrix to Python List of Lists */
PyObject* c_to_python_matrix(double **matrix, int rows, int cols) {
    int i, j;
    PyObject *outer_list = PyList_New(rows);
    for (i = 0; i < rows; i++) {
        PyObject *inner_list = PyList_New(cols);
        for (j = 0; j < cols; j++) {
            PyList_SetItem(inner_list, j, PyFloat_FromDouble(matrix[i][j]));
        }
        PyList_SetItem(outer_list, i, inner_list);
    }
    return outer_list;
}

/* 1. Wrapper for sym goal */
static PyObject* sym_wrapper(PyObject *self, PyObject *args) {
    PyObject *input_list;
    int n, d;
    if (!PyArg_ParseTuple(args, "Oii", &input_list, &n, &d)) return NULL;

    double **X = python_to_c_matrix(input_list, n, d);
    double **A = allocate_matrix(n, n);

    compute_sym(X, A, n, d);

    PyObject *result = c_to_python_matrix(A, n, n);
    free_matrix(X, n);
    free_matrix(A, n);
    return result;
}

/* 2. Wrapper for ddg goal */
static PyObject* ddg_wrapper(PyObject *self, PyObject *args) {
    PyObject *input_list;
    int n, d;
    if (!PyArg_ParseTuple(args, "Oii", &input_list, &n, &d)) return NULL;

    double **X = python_to_c_matrix(input_list, n, d);
    double **A = allocate_matrix(n, n);
    double **D = allocate_matrix(n, n);

    compute_sym(X, A, n, d);
    compute_ddg(A, D, n);

    PyObject *result = c_to_python_matrix(D, n, n);
    free_matrix(X, n);
    free_matrix(A, n);
    free_matrix(D, n);
    return result;
}

/* 3. Wrapper for norm goal */
static PyObject* norm_wrapper(PyObject *self, PyObject *args) {
    PyObject *input_list;
    int n, d;
    if (!PyArg_ParseTuple(args, "Oii", &input_list, &n, &d)) return NULL;

    double **X = python_to_c_matrix(input_list, n, d);
    double **A = allocate_matrix(n, n);
    double **D = allocate_matrix(n, n);
    double **W = allocate_matrix(n, n);

    compute_sym(X, A, n, d);
    compute_ddg(A, D, n);
    compute_norm(A, D, W, n);

    PyObject *result = c_to_python_matrix(W, n, n);
    free_matrix(X, n);
    free_matrix(A, n);
    free_matrix(D, n);
    free_matrix(W, n);
    return result;
}

/* 4. Wrapper for symnmf goal (Step 1.4.2 full algorithm) */
static PyObject* symnmf_wrapper(PyObject *self, PyObject *args) {
    PyObject *W_obj, *H_obj;
    int n, k;
    if (!PyArg_ParseTuple(args, "OOii", &W_obj, &H_obj, &n, &k)) return NULL;

    double **W = python_to_c_matrix(W_obj, n, n);
    double **H = python_to_c_matrix(H_obj, n, k);

    symnmf_algorithm(W, H, n, k);

    PyObject *result = c_to_python_matrix(H, n, k);
    free_matrix(W, n);
    free_matrix(H, n);
    return result;
}

/* Method Table */
static PyMethodDef symnmf_methods[] = {
    {"sym", (PyCFunction)sym_wrapper, METH_VARARGS, "Calculate Similarity Matrix"},
    {"ddg", (PyCFunction)ddg_wrapper, METH_VARARGS, "Calculate Diagonal Degree Matrix"},
    {"norm", (PyCFunction)norm_wrapper, METH_VARARGS, "Calculate Normalized Similarity Matrix"},
    {"symnmf", (PyCFunction)symnmf_wrapper, METH_VARARGS, "Perform full SymNMF H update"},
    {NULL, NULL, 0, NULL}
};

/* Module Definition */
static struct PyModuleDef symnmf_module = {
    PyModuleDef_HEAD_INIT,
    "symnmf",
    "C extension for SymNMF clustering algorithm",
    -1,
    symnmf_methods
};

/* Module Initialization */
PyMODINIT_FUNC PyInit_symnmf(void) {
    return PyModule_Create(&symnmf_module);
}