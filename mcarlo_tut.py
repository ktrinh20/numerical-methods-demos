'''
Introduction to Monte Carlo integration in Python

Name:  Kevin Trinh
Goal:  Integrate area of circle in first quadrant
'''

import math
import random as rnd

## GLOBAL VARIABLES ##
N = 10000  # number of random points

def f(x,y):
    if x**2 + y**2 > 1:
        return 0
    return 1

def mcarlo():
    f1_arr = []
    f2_arr = []
    for i in range(0, N):
        # generate random point in enclosing square
        xr = rnd.uniform(-1, 1)
        yr = rnd.uniform(-1, 1)
        # find <f> and <f^2>
        f1_arr.append(f(xr,yr))
        f2_arr.append(f(xr,yr)**2)
    f1 = sum(f1_arr) / N
    f2 = sum(f2_arr) / N
    
    # apply main formula
    I = 4*f1
    I_std = 4 * math.sqrt((f2 - f1**2)/N)
    
    return I, I_std
    
# run 5 experiments
for trial in range(0,5):
    I, I_std = mcarlo()
    print("Area:  " + str(I) + " +/- " + str(I_std))
    

