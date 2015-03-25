# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 10:48:19 2014

@author: Hayley
"""

import numpy as np
from scipy import stats
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pdb #pdb.set_trace() where want to set breakpoint and have debugging ability
import model_cython as fast
#%%
 
def return_summed_chisquare(params_Go, data):
    data150, data125, data100, data_onsets = data
    error_term = error_function_Go(params_Go, data150, data125, data100, data_onsets)
    #optGo = opt.minimize(error_function_Go, params_Go, args=(exp_MEPs_150, exp_MEPs_125, exp_MEPs_100, exp_EMG_onsets_three_stim), method='Nelder-Mead', tol=0.01)
    return error_term #optGo
    
    
def get_fac(t, params):
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
    k_facGo, tau_facGo, pre_t = params
    res = np.zeros_like(t)
    fac = 1/k_facGo * ((t + pre_t) - tau_facGo * (1 - np.exp(-(t + pre_t)/tau_facGo)))
    idx = (t + pre_t) >= 0
    res[idx] = fac[idx]
    return res
    
#%% 

def get_inhib_tonic(t, inhib):
    '''
    Generates a single horizontal line for tonic inhibition
    
    Parameters
    --------------
    t : array
        sequence of time index
    params : sequence (5,) of float
        inhib - height of inhibition line
        
    Returns
    ----------
    inhib_tonic : array
        array of same size as time index with constant inhibion value    
    '''
    #k_facGo, pre_t_mean, pre_t_sd, tau_facGo, inhib = params # , inhib_sd
    #inhib_value = np.random.normal(inhib, inhib_sd, size=1)
    inhib_tonic = np.ones(t.shape) * inhib # creates an array the same size as t, setting each element to 1, then multiplying by what value of inhib has come from normal distribution of inhib value being tested by function
    return inhib_tonic # returns array of 600 x inhib value as horizontal line for tonic inhib
    
#%%    
def get_inhib_increase(t, inhib_tonic, params_GS):
    '''
    
    '''
    k_inhib, tau_inhib, step_t_mean, step_t_sd = params_GS # step_t in this case refers to shifting along x axis where step input to tonic inhibition occurs
    threshold = np.zeros_like(t)    
    step_t = np.random.normal(step_t_mean, step_t_sd) # size=n_rep
    inhib = k_inhib * (1 - np.exp(-(t+step_t)/tau_inhib)) + inhib_tonic # by adding t + step_t, now only plotting inhib after intercept at zero
    inhib = np.maximum(inhib, inhib_tonic) # uses whichever value is greater from inhib or inhib_tonic - only shows the part of inhib curve that adds to tonic level after intercepted tonic value
    return inhib

#%%
def get_trials(params, n_rep=100000): 
    '''
    Generates n_rep number of facilitation curves for Go response for all simulated trials required
    
    Parameters
    -------------
    params : sequence (4,) of float
        k_facGo - scale of fac curve
        pre_t_mean - average start time before target presentation
        pre_t_sd - standard deviation of start time before target
        
        Returns
        --------
        fac_i : array
            facilitation curves for all simulated trials
        t : array
            sequence of time index
    '''
    k_facGo, pre_t_mean, pre_t_sd, tau_facGo, inhib_mean, inhib_sd = params  
    t = np.linspace(-.4, .2, 600, endpoint=False)  
#    tau_facGo = 2  # Currently set, but will need to optomize
    pre_t = np.random.normal(pre_t_mean, pre_t_sd, size=n_rep) # generates n_rep random numbers from a normal distribution of mean, sd that given into function
    fac_i = np.zeros((n_rep, t.size))  # had to change from fac_i, t - why does this cause error now?!?! sets up empty array of zeros for all simulated trials
    inhib_tonic = np.zeros((n_rep, t.size))    
    inhib = np.random.normal(inhib_mean, inhib_sd, size=n_rep)
    inhib_tonic += inhib[:,np.newaxis]
        
    for i in range(n_rep):  # for each simulated trial
        myparams_fac = k_facGo, tau_facGo, pre_t[i]  # takes parameters passed into model plus pre_t number randomly generated for that simulated trial
        fac_i[i] = fast.get_fac(t, myparams_fac)  # generates curve for that simulated trial
        #inhib_tonic[i] = get_inhib_tonic(t, inhib[i])
    return fac_i, inhib_tonic, t

#%% 
def get_activation_thresholds(t, inhib_tonic, params_GS, n_rep=100000):
    '''
    '''
    k_inhib, tau_inhib, step_t_mean, step_t_sd = params_GS
    thresholds = np.zeros((n_rep, t.size))
    for i in range(n_rep):
        thresholds[i] = get_inhib_increase(t, inhib_tonic[i], params_GS)
    return thresholds

#%%   
def get_fac_tms_vals(t, fac_i, pts=(-.15, -.125, -.1)):
    '''
    Gets values at pre-defined time points on all simulated fac curves
    
    Parameters 
    -------------
    pts : sequence of floats
        time points for comparison to MEP amplitude values from exp data
        
    Returns
    -------------
    vals : value on all simulated curves at time point requested
    '''
    #pdb.set_trace()
    idx150 = np.flatnonzero(np.isclose(t, pts[0]))
    vals150 = fac_i[:,idx150]
    idx125 = np.flatnonzero(np.isclose(t, pts[1]))
    vals125 = fac_i[:,idx125]
    idx100 = np.flatnonzero(np.isclose(t, pts[2]))
    vals100 = fac_i[:,idx100]
    return (vals150, vals125, vals100)  

#%%    
def get_emg_onsets(t, fac_i, inhib):
    '''
    '''
    getinhib = fac_i < inhib # for each curve, finds if true or false that value for fac_i is less than value for inhib
    switches = np.diff(getinhib) # diff function minuses each element from the previous one
    index_trials = np.nonzero(switches == 1) # finds indexes of all cases when values change from fac_i being below to above inhib value i.e. when curve crossing horizontal line
    return t[index_trials[1]] # finds actual time value at those indexes of curve intersection points

#%%  
def get_GS_tms_vals(t, go_curves, inhib_step, inhib_tonic, pts=(-0.075, -0.05, -0.025)):
    '''
    inhib_step:
    is activation threshold with step increase to tonic inhibition level
    '''
    index75 = np.flatnonzero(np.isclose(t, pts[0]))
    fac_values75 = go_curves[:, index75]
    inhib_step_values75 = inhib_step[:, index75]
    #pdb.set_trace()
    diff_inhib75 = inhib_step_values75 - inhib_tonic # inhib_tonic is single value for level of inhib
    pred75 = fac_values75 - diff_inhib75 #go_curves[:, index75] - (inhib_step[:, index75] - inhib_tonic[:, index75]) #diff_inhib75
    index50 = np.flatnonzero(np.isclose(t, pts[1]))
    fac_values50 = go_curves[:, index50]
    inhib_step_values50 = inhib_step[:, index50]
    diff_inhib50 = inhib_step_values50 - inhib_tonic
    pred50 = fac_values50 - diff_inhib50
    index25 = np.flatnonzero(np.isclose(t, pts[2]))
    fac_values25 = go_curves[:, index25]
    inhib_step_values25 = inhib_step[:, index25]
    diff_inhib25 = inhib_step_values25 - inhib_tonic
    pred25 = fac_values25 - diff_inhib25    
    return pred75, pred50, pred25
    
#%%
def get_chisquare(obs_data, obs_model, nbins=3):
    '''
    Sends into function actual MEP amplitude data at each time point, predicted
    amplitudes from simulated facilitation curves, number of bins to divide
    percentile bins into. Calculates histograms for experimental and predicted
    data. Compares frequencies in each bin. Calculates one-way chi square test.
    
    Parameters
    --------------
    obs_data : array of MEP amplitudes from experimental data
    obs_model : predicted MEP amplitudes from simulated trials (i.e. facilitation curves)
    
    Returns
    ---------------
    chi square statistic for how well predicted data matches experimental data
    
    '''
    percentile_bins = np.linspace(0, 100, nbins + 1)    
    bin_edges = np.percentile(obs_data, list(percentile_bins))
    hist_data, bin_edges  = np.histogram(obs_data,  bins=bin_edges)
    hist_data = hist_data / float(obs_data.size) # still presents frequencies proportional to number of observations? - check with Angus 
    # put in density so that value for each bin is expressed as proportion of total number of observations
    hist_model, bin_edges = np.histogram(obs_model, bins=bin_edges)
    hist_model = hist_model / float(obs_model.size)
    return stats.chisquare(hist_data, hist_model)

#%%
def load_exp_data(fname):
    file_contents = np.genfromtxt(fname, dtype=float, delimiter=",", skiprows=1)  # loads the csv file as a float, skipping the first row as only subject codes
    MEP_amps_mV  = file_contents.flatten()
    no_nan_MEP_amps_mV = MEP_amps_mV[~np.isnan(MEP_amps_mV)] # Creates array of True False for whether is NaN - then indexes out of MEP_amps_mV array only with corresponding returned True
    return no_nan_MEP_amps_mV

#%%
def error_function_Go(params, data150, data125, data100, data_onsets):  
    print "Trying with values: " + str(params) 
    fac_i, inhib_tonic, t = get_trials(params)  # now have 6 parameters - fifth param is now inhib value, sixth is inhib sd
#    inhib_tonic = get_inhib_tonic(t, params) 
    pred150, pred125, pred100 = get_fac_tms_vals(t, fac_i)    
    pred_onsets = get_emg_onsets(t, fac_i, inhib_tonic) 
    X2_onsets = get_chisquare(data_onsets, pred_onsets, nbins=2)[0]
    print "X2_onsets: ", X2_onsets
    X2_150 = get_chisquare(data150, pred150, nbins=2)[0]
    print "X2_150: ", X2_150
    X2_125 = get_chisquare(data125, pred125, nbins=2)[0]
    print "X2_125: ", X2_125
    X2_100 = get_chisquare(data100, pred100, nbins=2)[0]
    print "X2_100: ", X2_100
    X2_summed_Go = X2_150 + X2_125 + X2_100 + X2_onsets
    print "X2 summed: ", X2_summed_Go
    return X2_summed_Go
     

#%%
def error_function_GS(params_GS, params_Go, data75, data50, data25, data_onsets): # can I pass it two lots of parameter lists?
    print "Trying with values: " + str(params_GS)
    fac_i, inhib_tonic, t = params_Go
    #inhib_tonic = get_inhib_tonic(t, params_Go)
    activation_thresholds = get_activation_thresholds(t, inhib_tonic, params_GS) # generates activation threshold with step increase to tonic inhib level
    pred75, pred50, pred25 = get_GS_tms_vals(t, fac_i, activation_thresholds, inhib_tonic)
#    pred_onsets = get_emg_onsets(t, fac_i, activation_thresholds)
    X2_75 = get_chisquare(data75, pred75, nbins=2)[0]
    print "X2_75: ", X2_75
    X2_50 = get_chisquare(data50, pred50, nbins=2)[0]
    print "X2_50: ", X2_50
    X2_25 = get_chisquare(data25, pred25, nbins=2)[0]
    print "X2_25: ", X2_25
    X2_summed_GS = X2_75 + X2_50 + X2_25
    print "X2_summed: ", X2_summed_GS
    return X2_summed_GS
    

#%%
    # Load data
data_dir = 'C:\Users\Hayley\Documents\University\PhD\PhD\Modeling\Experimental data for model\Using'
#data_dir = ''
# Loading experimental data for Go trials 
# MEP data
fname150 = data_dir + '\Go_trial_MEP_amplitudes_150ms.csv'
fname125 = data_dir + '\Go_trial_MEP_amplitudes_125ms.csv'
fname100 = data_dir + '\Go_trial_MEP_amplitudes_100ms.csv'
exp_MEPs_150 = load_exp_data(fname150)
exp_MEPs_125 = load_exp_data(fname125)
exp_MEPs_100 = load_exp_data(fname100)
# EMG onsets
fnameGoThreeStimOnly = data_dir + '\Go_EMG onsets_only 3 stim times.csv'
exp_EMG_onsets_three_stim = load_exp_data(fnameGoThreeStimOnly) / 1000 - .8 # # Uses same load_exp_data function as for MEP data, saving output variable as EMG onset. /1000 to put into sectonds, -0.8 to set relative to target line at 0ms

data_Go = exp_MEPs_150, exp_MEPs_125, exp_MEPs_100, exp_EMG_onsets_three_stim

# Loading experimental data for Go Left - Stop Right (GS) trials 
 #MEP data
fnameGS75 = data_dir + '\MEP data_GS trials_75ms.csv'
fnameGS50 = data_dir + '\MEP data_GS trials_50ms.csv'
fnameGS25 = data_dir + '\MEP data_GS trials_25ms.csv'
exp_GS_MEPs_75 = load_exp_data(fnameGS75)
exp_GS_MEPs_50 = load_exp_data(fnameGS50)
exp_GS_MEPs_25 = load_exp_data(fnameGS25)
 #EMG onsets
fnameGSStimOnsets = data_dir + '\GS_EMG onsets_3 stim times.csv'
exp_GS_EMG_onsets_three_stim = load_exp_data(fnameGSStimOnsets) / 1000 - .8

data_GS = exp_GS_MEPs_75, exp_GS_MEPs_50, exp_GS_MEPs_25, exp_GS_EMG_onsets_three_stim
    
#%%
 #optomizing parameters for Go trial baseline facilitation curve and tonic inhibition level
if __name__ == "__main__":
    params_Go = [0.004, 0.2, 0.04, 2, 1.6, 1.0] # values for k_facGo, pre_t_mean, pre_t_sd, tau_facGo, inhib_tonic, inhib_sd - old starting point [0.06, 0.4, 0.1, 2, 1, 0.2]
    optGo = opt.minimize(error_function_Go, params_Go, args=(exp_MEPs_150, exp_MEPs_125, exp_MEPs_100, exp_EMG_onsets_three_stim), method='Nelder-Mead', tol=0.01) # trying tolerance to 3 dp. method="SLSQP", bounds=[(0,None),(0,None),(0,None),(None,None)])  
    print "ParamsOptimizedGo", optGo # returns array of parameter values when optimization terminated successfully
#    optGo = opt.fmin(error_function_Go, params_Go, args=(exp_MEPs_150, exp_MEPs_125, exp_MEPs_100, exp_EMG_onsets_three_stim), xtol=0.001, ftol=0.01) # testing scipy.optimize.fmin to set tolerances
 #optomizing parameters for GS trial activation threshold and single-component facilitation curve    
    params_Go = optGo.x #[0.004, 0.19, 0.02, 1.45, 1.61, 0.14] # output from Go optimization function 
    fac_i, inhib_tonic, t = get_trials(params_Go) # generate baseline Go fac curves from parameters already optimized
    components_Go = (fac_i, inhib_tonic, t)
    params_GS = [1.2, 0.8, 0.1, 0.02] # [1.2, 0.8, 0.2, 0.02] values for k_inhib, tau_inhib, step_t_mean, step_t_sd    
    optGS  = opt.minimize(error_function_GS, params_GS, args=(components_Go, exp_GS_MEPs_75, exp_GS_MEPs_50, exp_GS_MEPs_25, exp_GS_EMG_onsets_three_stim), method='Nelder-Mead', tol=0.01)    
    print "ParamsOptimizedGS", optGS
    
# optGS  = opt.minimize(error_function_GS, params_GS, args=(params0, exp_GS_MEPs_75, exp_GS_MEPs_50, exp_GS_MEPs_25), method='Nelder-Mead')    

#%%    
    # Values for params_Go and params_GS should come from cluster output
def visualize_params(params_Go, data_Go):  # visualizes fac curves and tonic inhibition on Go trials
    mep150, mep125, mep100, emg_onset = data_Go
    fac_i, inhib_tonic, t = get_trials(params_Go, n_rep=100)
    plt.plot(t, fac_i.T, 'k-', alpha=0.4)
    plt.plot(np.ones_like(mep150) * -0.15,  mep150, 'rx')
    plt.plot(np.ones_like(mep125) * -0.125, mep125, 'rx')
    plt.plot(np.ones_like(mep100) * -0.100, mep100, 'rx')
    #inhib_tonic = params_Go[-2]
    plt.plot(t, inhib_tonic.T, color='r')
    plt.plot(emg_onset, np.zeros_like(emg_onset), 'rx') #* inhib_tonic
#%%    
def visualize_params_GS(params_Go, params_GS, data_Go, data_GS):  # visualizes fac and inhibition on GS trials
    mep150, mep125, mep100, emg_onset_Go = data_Go
    mep75, mep50, mep25, emg_onset_GS = data_GS
    fac_i, inhib_tonic, t = get_trials(params_Go, n_rep=100)
    activation_thresholds = get_activation_thresholds(t, inhib_tonic, params_GS, n_rep=100)
    #fig, ax = plt.subplots()
    plt.plot(t, fac_i.T, 'k-', alpha=0.4)
    plt.plot(t, activation_thresholds.T, color='r')
    plt.plot(np.ones_like(mep150) * -0.15,  mep150, 'rx')
    plt.plot(np.ones_like(mep125) * -0.125, mep125, 'rx')
    plt.plot(np.ones_like(mep100) * -0.100, mep100, 'rx')
    plt.plot(np.ones_like(mep75) * -0.075,  mep75, 'rx')
    plt.plot(np.ones_like(mep50) * -0.05, mep50, 'rx')
    plt.plot(np.ones_like(mep25) * -0.025, mep25, 'rx')
#    inhib_tonic = params_Go[-2]
    #plt.axhline(inhib_tonic, color='r')
    plt.plot(emg_onset_GS, np.zeros_like(emg_onset_GS), 'rx') #* params_Go[-2]
    #plt.plot(emg_onset_Go, np.zeros_like(emg_onset_Go) * params_Go[-2], 'rx')
    
    