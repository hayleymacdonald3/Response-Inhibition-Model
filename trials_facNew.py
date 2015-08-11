# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 10:36:24 2015

Cuda code from Sina modified for new facilitation curve
"""

import numpy as np
from timeit import default_timer as time
import sys
from numbapro import cuda
from math import exp

DEFAULT_TRIALS=1e5

# CUDA version
@cuda.jit('void(float64[:,:], float64[:,:], int32, float64[:], int32, float64[:], float64[:], float64[:])')  # added another float64[:,:] for fac_Biman by Hayley
def _gaussian_cuda64(fac, fac_Biman, n_rep, t, n_t, a_facGo, b_facGo, c_facGo):
    i, j = cuda.grid(2)
    if i >= n_rep or j >= n_t:
        return

    # Fill in 2D fac data structure
    fac[i, j] = fac_Biman[i, j] + (a_facGo[i] * exp(-(t[j] - b_facGo[i])**2 /(2 * c_facGo[i]**2))) # now adding two arrays together for new fac curve by Hayley
 # original equation from Sina for single fac curve: fac[i, j] = a_facGo[i] * exp(-(t[j] - b_facGo[i])**2 /(2 * c_facGo[i]**2))



# CUDA version
@cuda.jit('void(float32[:,:], float32[:,:], int32, float32[:], int32, float32[:], float32[:], float32[:])')  # added another float32[:,:] for fac_Biman by Hayley
def _gaussian_cuda32(fac, fac_Biman, n_rep, t, n_t, a_facGo, b_facGo, c_facGo):
    i, j = cuda.grid(2)
    if i >= n_rep or j >= n_t:
        return

    # Fill in 2D fac data structure
    fac[i, j] = fac_Biman[i, j] + (a_facGo[i] * exp(-(t[j] - b_facGo[i])**2 /(2 * c_facGo[i]**2))) # now adding two arrays together for new fac curve by Hayley
# original equation from Sina for single fac curve: fac[i, j] = a_facGo[i] * exp(-(t[j] - b_facGo[i])**2 /(2 * c_facGo[i]**2))

# Non-parallel version but restructured from _gaussian_serial to form a basis
# for parallelisation
# Much slower because it avoid numpy vector operations
# renamed get_fac to _gaussian_parallel_base
def _gaussian_parallel_base(fac, n_rep, t, n_t, a_facGo, b_facGo, c_facGo):
    # for that simulated trial
    for i in range(n_rep):
        # generates curve for that simulated trial
        for j in range(n_t):
            # TODO save these to avoid ** repeats across i
            fac[i][j] = a_facGo[i] * np.exp(-(t[j] - b_facGo[i])**2 /(2 * c_facGo[i]**2))


# Original version
# renamed get_fac to _gaussian_original
def _gaussian_original(t, params, params_Bimanual): # added argument params_Bimanual by Hayley
    '''
    Generates a Gaussian facilitation curve

    Parameters
    ----------
    t : array
        sequence of time index
    params : sequence (3,) of float
        a_facGo - amplitude of Gaussian fac curve
        b_facGo - time to peak of fac curve
        c_facGo - curvature of fac curve

    Returns
    -------
    res : array
        fac curve values at times `t`
    '''
    fac_Biman = params_Bimanual # added by Hayley
    a_facGo, b_facGo, c_facGo = params
#    res = np.zeros_like(t)
    fac = np.add(fac_Biman, (a_facGo * np.exp(-(t - b_facGo)**2 /(2 * c_facGo**2)))) # modified equation so addition of two fac curves by Hayley
#    idx = (t + pre_t) >= 0
#    res[idx] = fac[idx]
    return fac


# Original version
# Renamed get_trials to gaussian_original
def gaussian_original(params, n_rep=DEFAULT_TRIALS, rec_time=False):
    '''
    Generates n_rep number of Guassian facilitation curves for Go response for all
    simulated trials required

    Parameters
    -------------
    params : sequence (4,) of float
        a_facGo - amplitude of gaussian curve
        b_facGo - time to peak of gaussian curve
        c_facGo - curvature of gaussian curve

        Returns
        --------
        fac_i : array
            facilitation curves for all simulated trials
        t : array
            sequence of time index
    '''

    # expand params
    a_facGo_mean, a_facGo_sd, b_facGo_mean, b_facGo_sd, c_facGo_mean, c_facGo_sd, \
    inhib_mean, inhib_sd = params

    t = np.linspace(-.4, .2, 600, endpoint=False)
    #tau_facGo = 2  # Currently set, but will need to optomize

    # generates n_rep random numbers from a normal distribution of mean, sd that given into function
    a_facGo = np.random.normal(a_facGo_mean, a_facGo_sd, size=n_rep)
    b_facGo = np.random.normal(b_facGo_mean, b_facGo_sd, size=n_rep)
    c_facGo = np.random.normal(c_facGo_mean, c_facGo_sd, size=n_rep)


    # had to change from fac_i, t - why does this cause error now?!?!
    # sets up empty array of zeros for all simulated trials
    fac_i = np.zeros((n_rep, t.size))


    inhib_tonic = np.zeros((n_rep, t.size))
    inhib = np.random.normal(inhib_mean, inhib_sd, size=n_rep)
    inhib_tonic += inhib[:,np.newaxis]

    # time performance if required
    if (rec_time):
        t_start = time()

    for i in range(n_rep):  # for each simulated trial
        # takes parameters passed into model plus pre_t number randomly generated
        # for that simulated trial
        myparams_fac = a_facGo[i], b_facGo[i], c_facGo[i]
        fac_i[i] = _gaussian_original(t, myparams_fac)  # generates curve for that simulated trial
        #inhib_tonic[i] = get_inhib_tonic(t, inhib[i])

    if (rec_time):
        t_diff = time() - t_start
        tps = n_rep / t_diff
        print "Original trials per second: %.0f" % (tps)

    return fac_i, inhib_tonic, t

# Get Gaussian trials using a give method with option to compare methods
def gaussian(method, params, fac_Bimanual, n_rep=DEFAULT_TRIALS, compare=None, dtype=np.float32): # added new argument fac_Bimanual by Hayley

    """("cuda", params_facNew, components_Go[0], n_rep=100000, dtype=np.float32)
    method, compare can be : "original", "parallel_base", "cuda"
    dtype float32 is faster for GPU but has lower precision
    dtype float64 is faster for CPU
    """

    # expand params - modified so now only parametizing b by Hayley
    a_facGo_mean = 2.6
    a_facGo_sd = 0.03
    b_facGo_mean, b_facGo_sd = params
    c_facGo_mean = 0.06
    c_facGo_sd = 0.01
#    a_facGo_mean, a_facGo_sd, b_facGo_mean, b_facGo_sd, c_facGo_mean, c_facGo_sd, \
#    inhib_mean, inhib_sd = params

    # create data structures
    t = np.linspace(-.4, .2, 600, endpoint=False).astype(dtype)
    #tau_facGo = 2  # Currently set, but will need to optomize

    # generates n_rep random numbers from a normal distribution of mean, sd that given into function
    a_facGo = np.random.normal(a_facGo_mean, a_facGo_sd, size=n_rep).astype(dtype)
    b_facGo = np.random.normal(b_facGo_mean, b_facGo_sd, size=n_rep).astype(dtype)
    c_facGo = np.random.normal(c_facGo_mean, c_facGo_sd, size=n_rep).astype(dtype)

#    inhib_tonic = np.zeros((n_rep, t.size))
#    inhib = np.random.normal(inhib_mean, inhib_sd, size=n_rep)
#    inhib_tonic += inhib[:,np.newaxis]

    # sets up empty array of zeros for all simulated trials
    fac1 = np.zeros((n_rep, t.size)).astype(dtype)
    facs = [fac1]
    if compare:
        fac2 = np.zeros((n_rep, t.size)).astype(dtype)
        facs = facs + [fac2]


    # Execute trials and compare performance and results if required
    tps = [0, 0] # trials per second for each method
    for fi, f in enumerate([method, compare]):
        if f: # check if method or comapre is not None
            fac = facs[fi] # get the right fac
            t_start = time()

            if (f == "original"):
                for i in range(n_rep):  # for each simulated trial
                    myparams_fac = a_facGo[i], b_facGo[i], c_facGo[i]
                     # generates curve for that simulated trial
                    fac[i] = _gaussian_original(t, myparams_fac, fac_Bimanual[i]) # added fac_Bimanual[i] by Hayley
            elif (f == "parallel_base"):
                _gaussian_parallel_base(fac, n_rep, t, len(t), a_facGo, b_facGo, c_facGo)
            elif (f == "cuda"):
                # Setup CUDA variables
                tpb_x = 8 # threads per block in x dimension
                tpb_y = 8 # threads per block in y dimension
                block_dim = tpb_x, tpb_y
                bpg_x = int(n_rep / tpb_x) + 1 # block grid x dimension
                bpg_y = int(t.size / tpb_y) + 1 # block grid y dimension
                grid_dim = bpg_x, bpg_y

                stream = cuda.stream()
                with stream.auto_synchronize():
                    d_fac = cuda.to_device(fac, stream)
                    d_fac_Biman = cuda.to_device(fac_Bimanual, stream) # added by Hayley
                    d_t = cuda.to_device(t, stream)
                    d_a_facGo = cuda.to_device(a_facGo, stream)
                    d_b_facGo = cuda.to_device(b_facGo, stream)
                    d_c_facGo = cuda.to_device(c_facGo, stream)
                    #print "CUDA kernel: Block dim: ({tx}, {ty}), Grid dim: ({gx}, {gy})".format(tx=tpb_x, ty=tpb_y, gx=bpg_x, gy=bpg_y)
                    if dtype == np.float32:
                        _gaussian_cuda32[grid_dim, block_dim](d_fac, d_fac_Biman, n_rep, d_t, len(t), d_a_facGo, d_b_facGo, d_c_facGo) # added argument d_fac_Biman by Hayley
                    elif dtype == np.float64:
                        _gaussian_cuda64[grid_dim, block_dim](d_fac, d_fac_Biman, n_rep, d_t, len(t), d_a_facGo, d_b_facGo, d_c_facGo) # added argument d_fac_Biman by Hayley
                    else:
                        print "Error: CUDA dtype must be np.float32 or np.float64"
                        sys.exit(1)
                    d_fac.to_host(stream)

            t_diff = time() - t_start
            tps[fi] = n_rep / t_diff

    # Check results close enough
    if compare:
        close = np.allclose(facs[0], facs[1], rtol=0, atol=1e-05)
        if not close:
            print "ERROR: results from method '%s' are not the same as method '%s'" % (method, compare)
            #print (facs[1] - facs[0])
            sys.exit(1);


    # Summary
    print "%s trials per second: %.0f" % (method, tps[0])
    if compare:
        print "%s trials per second: %.0f" % (compare, tps[1])
        print "Speed up: %.3f x" %(tps[0]/tps[1]) # method / compare
        print "Results close enough? ", close

    return fac1 #, inhib_tonic, t