# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 20:04:22 2015

@author: Hayley
"""
import Go_trials

params_Go = [0.008, 0.2, 0.05, 2, 1.5, 0.002] # want these all initially set at those for suspected global minimum i.e. best optimization

error_term = Go_trials.return_summed_chisquare(params_Go)
print 'Look here'
print error_term
