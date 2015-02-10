'''
build this file by calling:

python setup.py build_ext --inplace
'''

import numpy as np

cimport numpy as np

DTYPEt = np.float64
ctypedef np.float64_t DTYPE_t

DTYPEb = np.int8
ctypedef np.int8_t DTYPE_b

cdef extern from "math.h":
    double exp(double x)

def get_fac(np.ndarray[DTYPE_t, ndim=1] t, params):
    '''
    Generates a facilitation curve
    
    Parameters
    ----------
    t : array
        sequence of time index
    params : sequence (3,) of float
        k_facGo - scale of fac curve
        tau_facGo - curvature of fac curve
        pre_t - start time before target presentation
        
    Returns
    -------
    res : array
        fac curve values at times `t`

    '''
    cdef float k_facGo = params[0]
    cdef float tau_facGo = params[1]
    cdef float pre_t = params[2]
    cdef np.ndarray[np.float64_t, ndim=1] res = np.zeros_like(t)
    cdef float temp
    cdef int i
    for i in xrange(len(t)):
        temp =  1/k_facGo * ((t[i] + pre_t) - tau_facGo * (1 - exp(-(t[i] + pre_t)/tau_facGo)))
        if ((t[i] + pre_t) >= 0):
            res[i] = temp
    return res

def get_trials(np.ndarray[DTYPE_t, ndim=1] t, params, int n_rep=10000):
    '''
    Generates n_rep number of facilitation curves for Go response for all simulated trials required
    
    Parameters
    -------------
    params : sequence (4,) of float
        k_facGo - scale of fac curve
        pre_t_mean - average start time before target presentation
        pre_t_sd - standard deviation of start time before target
        tau_facGo - time constant of facilitation curve
        
        Returns
        --------
        fac_i : array
            facilitation curves for all simulated trials
        t : array
            sequence of time index
    '''
    cdef float k_facGo = params[0]
    cdef float pre_t_mean = params[1]
    cdef float pre_t_sd = params[2]
    cdef float tau_facGo = params[3]
    
    cdef int i, j
    cdef np.ndarray[DTYPE_t, ndim=2]res = np.zeros((n_rep, t.size))
    cdef float temp, pre_t
    for i in xrange(n_rep):
        pre_t = np.random.normal(pre_t_mean, pre_t_sd, size=1) # generates 1 random # from normal dist.
        for j in xrange(len(t)):
            temp =  1/k_facGo * ((t[j] + pre_t) - tau_facGo * (1 - exp(-(t[j] + pre_t)/tau_facGo)))
            if ((t[j] + pre_t) >= 0):
                res[i,j] = temp
    return res
