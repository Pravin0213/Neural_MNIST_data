# -*- coding: utf-8 -*-
"""
Created on Sun May 18 12:37:39 2025

@author: praveenreddy.bobbali
"""

import pickle
with open('weights.pkl', 'rb') as f:
    weights = pickle.load(f)
with open('biases.pkl', 'rb') as f:
    biases = pickle.load(f)
 
    
a =weights
b=biases
print(weights, biases)