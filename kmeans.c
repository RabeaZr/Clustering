#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdio.h>
#include <stdlib.h>

// kmeans.c is the file where we implemented both CAPI and the kmeans algorithm
// the first function that python calls is the wrapping function (c_means_capi) which then calls other functions
// in order to implement the k means algorithm, we use pyobjects in order to make sure that we return to python
// a pyobject

PyObject* PyList_New(Py_ssize_t Len);
int PyList_SetItem(PyObject * list, Py_ssize_t index,PyObject * item);

static double distance(const double *x, const double *y, int D){
    // a fucntion that gets two arrays of doubles and their length and return the distance between them
    double sum = 0;
    int i;
    for(i = 0; i < D; i++)
    {
        sum = sum + (x[i]-y[i])*(x[i]-y[i]);
    }
    return sum;
}

static void copyArray(double **x, double **y, int rows, int cols){
    // a function that takes two matrices and their dimensions copies the first one into the second one
    int i;
    int j;
    for(i = 0; i<rows;i++)
    {
        for(j=0;j<cols;j++)
        {
            y[i][j]=x[i][j];
        }
    }
}

static void Add(double *centroid, double *x, int D){
    // a function that takes two arrays and their length and adds them together and  saves the result in the first one
    int i;
    for(i = 0 ; i < D ; i++){
        centroid[i] = centroid[i] + x[i];
    }
}

static void Div(double *centroid, int size, int D){
    // a function that takes an array and a number and it divides each entry of the array by that number
    int i;
    for(i = 0; i <  D; i++){
        centroid[i] = centroid[i]/(double)size;
    }
}

static int EqualArray(double **x, double **y,int K,int D){
    // a function that takes two matrices and their dimensions and checks if the matrices are equal or not
    // it just runs over every entry and checks if both matrices agree on it or not
    int i;
    int j;
    for(i = 0; i < K; i++)
    {
        for(j = 0; j < D; j++){
            if(x[i][j] != y[i][j]){
                return 0;
            }
        }
    }
    return 1;
}

static void resetArray(double **x, int K, int D)
    // this functions gets a matrix and its dimensions and resets it to zeros
{
    int i;
    int j;
    for(i = 0;i<K;i++)
    {
        for(j=0;j<D;j++)
        {
            x[i][j]=0;
        }
    }
}

static void NewCentroids(double **array,double **centroids,int *mapXItoCluster, int *ClusterSizes, int N, int K, int D)
{   // after we chose an observation to add to a cluster we have to update the centroids, this function takes care of that
    int i;
    int j;
    resetArray(centroids,K,D);
    for(i=0;i<N;i++){
        Add(centroids[mapXItoCluster[i]], array[i], D);
    }
    for(j = 0 ; j < K;j++){
        Div(centroids[j],ClusterSizes[j],D);
    }
}
static int * c_means(int N,int K ,int D,int MAX_ITER,double** array,double** centroids) {
    // this function is basically k means algorithm that we implemented in the first project, this function
    // gets the amount of observations, clusters, dimension of each observation, maximum iterations, the observations
    // and the initial centroids
    int i;
    int j;
    int z;
    double dis;
    double mindis;
    int mindisvec;
    int flag = 1;
    int *mapXItoCluster;
    int *ClusterSizes;
    double **centroidsCopy;

    centroidsCopy = malloc(K*sizeof(double*));
    if(centroidsCopy == NULL)
    {
        return NULL;
    }

    for(i = 0 ; i < K ; i++){
        centroidsCopy[i] = malloc(D*sizeof(double));
        if(centroidsCopy[i] == NULL)
        {
            for(j=0;j<i;j++)
            {
                free(centroidsCopy[j]);
            }
            free(centroidsCopy);
            return NULL;
        }
    }

    z=0;
    mindisvec = 0;
    mapXItoCluster = malloc(N*sizeof(int));

    if(mapXItoCluster == NULL)
    {
        for(j=0;j<K;j++)
        {
            free(centroidsCopy[j]);
        }
        free(centroidsCopy);
        return NULL;
    }

    ClusterSizes = malloc(K*sizeof(int));


    if(ClusterSizes == NULL)
    {
        for(j=0;j<K;j++)
        {
            free(centroidsCopy[j]);
        }
        free(centroidsCopy);
        free(mapXItoCluster);
        return NULL;
    }

    while((z<MAX_ITER)&&(flag==1))
    {
        for(i = 0 ; i < K; i++){
            ClusterSizes[i] = 0;
        }
        for(i=0;i<N;i++){
            mindis = INFINITY;
            for(j=0;j<K;j++)
            {
                dis=distance(array[i],centroids[j],D);
                if(dis<mindis)
                {
                    mindis=dis;
                    mindisvec=j;
                }
            }
            mapXItoCluster[i]=mindisvec;
            ClusterSizes[mindisvec]++;
        }
        copyArray(centroids,centroidsCopy,K,D);
        NewCentroids(array,centroids,mapXItoCluster,ClusterSizes,N,K,D);
        if(EqualArray(centroids,centroidsCopy,K,D) == 1){
            flag = 0;
        }
        z++;
    }

    for(i = 0 ; i < K ; i++){
        free(centroidsCopy[i]);
    }
    free(centroidsCopy);
    free(ClusterSizes);
    return mapXItoCluster;
}

static PyObject* c_means_capi(PyObject *self, PyObject *args)
{   // this function is the function that connects between python and C, this is also a wrapping function for cmeans
    // which is the function above (where we implemented k means)
    int MAX_ITER;
    double **array, **centroids;
    PyObject  *_listOfObs, *_listOfCent, *arrRow, *centRow, *koko;
    Py_ssize_t a, b, c, i, j;
    long N;
    long K;
    long D;
    if(!PyArg_ParseTuple(args, "iOO", &MAX_ITER ,&_listOfObs, &_listOfCent)) {
        return NULL;
    }

    if (!PyList_Check(_listOfObs) || !PyList_Check(_listOfCent))
        return NULL;


    N = (long) PyObject_Length(_listOfObs);
    K = (long) PyObject_Length(_listOfCent);
    D = (long) PyObject_Length(PyList_GetItem(_listOfObs,0));

    a = PyList_Size(_listOfObs);
    b = PyList_Size(_listOfCent);
    c = PyList_Size(PyList_GetItem(_listOfObs,0));


    array = malloc(sizeof(double *) * N);
    if(array == NULL)
    {
        printf("memory allocation error");
        PyErr_SetString(PyExc_MemoryError, "error while creating the c module");
        return NULL;
    }

    for (i = 0; i < N; ++i) {
        array[i] = malloc(sizeof(double) * D);
        if(array[i] == NULL)
        {
            for(j = 0; j<i; j++)
            {
                free(array[j]);
            }
            free(array);
            printf("memory allocation error");
            PyErr_SetString(PyExc_MemoryError, "error while creating the c module");
            return NULL;

        }
    }
    centroids = malloc(sizeof(double *) * K);
    if(centroids == NULL)
    {
        for(i=0;i<N;i++)
        {
            free(array[i]);
        }
        free(array);
        printf("memory allocation error");
        PyErr_SetString(PyExc_MemoryError, "error while creating the c module");
        return NULL;
    }

    for (i = 0; i < K; ++i) {
        centroids[i] = malloc(sizeof(double) * D);
        if(centroids[i] == NULL)
        {
            for(j=0;j<N;j++)
            {
                free(array[j]);
            }
            free(array);
            for(j=0;j<i;j++)
            {
                free(centroids[j]);
            }
            free(centroids);
            printf("memory allocation error");
            PyErr_SetString(PyExc_MemoryError, "error while creating the c module");
            return NULL;
        }
    }

    for (i = 0; i < b; i++) {
        arrRow = PyList_GetItem(_listOfObs, i);
        centRow = PyList_GetItem(_listOfCent, i);
        for(j = 0; j < c; j++){
            koko = PyList_GetItem(arrRow, j);
            array[i][j] = PyFloat_AsDouble(koko);
            koko = PyList_GetItem(centRow, j);
            centroids[i][j] = PyFloat_AsDouble(koko);
        }
    }

    for (i = b; i < a; i++) {
        arrRow = PyList_GetItem(_listOfObs, i);
        for(j = 0; j < c; j++){
            koko = PyList_GetItem(arrRow, j);
            array[i][j] = PyFloat_AsDouble(koko);
        }
    }
    int * XImapping = c_means(N, K, D, MAX_ITER,array,centroids);

    for (i = 0 ; i < N  ; i++){
        free(array[i]);
    }
    free(array);
    for (j = 0 ; j < K ; j++){
        free(centroids[j]);
    }
    free(centroids);

    if (XImapping == NULL){
        printf("memory allocation error");
        PyErr_SetString(PyExc_MemoryError, "error while creating the c module");
        return NULL;
    }

    PyObject * python_res = PyList_New(N);
    PyObject * temp;
    for (i = 0; i < N; i++){
        temp = Py_BuildValue("i", XImapping[i]);
        PyList_SetItem(python_res,i,temp);
    }
    return python_res;
}

// defining the name of the function c_means_capi to be kmeans, in order to use it python with that name
static PyMethodDef _methods[] = {
    {"kmeans", (PyCFunction)c_means_capi, METH_VARARGS, PyDoc_STR("Please enter the fields : MAX_ITER and arrays of : Observations and Centroids")},
    {NULL, NULL, 0, NULL}
};

//defining the name of the module
static struct PyModuleDef _moduledef = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    NULL,
    -1,
    _methods
};

PyMODINIT_FUNC
// initializing the CAPI module
PyInit_mykmeanssp(void)
{
    PyObject *m;
    m = PyModule_Create(&_moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}