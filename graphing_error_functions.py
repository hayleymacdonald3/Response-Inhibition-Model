# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 20:04:22 2015

@author: Hayley
"""
import Go_trials
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def load_exp_data(fname):
    file_contents = np.genfromtxt(fname, dtype=float, delimiter=",", skiprows=1)  # loads the csv file as a float, skipping the first row as only subject codes
    MEP_amps_mV  = file_contents.flatten()
    no_nan_MEP_amps_mV = MEP_amps_mV[~np.isnan(MEP_amps_mV)] # Creates array of True False for whether is NaN - then indexes out of MEP_amps_mV array only with corresponding returned True
    return no_nan_MEP_amps_mV

    # Load data
data_dir = 'C:\Users\Hayley\Documents\University\PhD\PhD\Modeling\Experimental data for model'
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

data = exp_MEPs_150, exp_MEPs_125, exp_MEPs_100, exp_EMG_onsets_three_stim

# k_facGo, pre_t_mean, pre_t_sd, tau_facGo, inhib_tonic, inhib_sd = parameter values
params_Go = [0.004, 0.19, 0.02, 1.45, 1.61, 0.14] # gives summed chisquare of 0.03 - set at those for suspected global minimum i.e. best optimization
range_for_parameter = 0.51 # need to set this one increment higher than actual value you want as end of range
increment_for_parameter = 0.01
error_term = np.zeros(range_for_parameter/increment_for_parameter - 1)
error_term_index = 0
x_axis = np.zeros(range_for_parameter/increment_for_parameter - 1)

def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step
        
for i in drange(increment_for_parameter, range_for_parameter, increment_for_parameter):  
    params_Go[1] = i  # index will change depending on which parameter investigating
    x_axis[error_term_index] = i
    error_term[error_term_index] = Go_trials.return_summed_chisquare(params_Go, data)
    error_term_index = error_term_index + 1
    
for i in range(error_term_index):
    if error_term[i] > 10:
        error_term[i] = 10

plt.plot(x_axis, error_term, alpha=0.5)

