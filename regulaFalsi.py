'''
Introduction to Secant Methods in Python -- Regula Falsi

Name: Kevin Trinh
Goal: Find the root of log(3x/2)
'''

import math
import scipy as sp

def func(x):
    '''The function that we are finding the root of.'''
    return sp.log(1.5 * x)

def regulaFalsi(a, b, tol=1e-15, maxiter=1000):
    '''Perform Regula Falsi root-finding algorithm. Uses f(x) in the tolerance criterion.'''
    
    # check that the root is bracketed
    assert func(a) * func(b) < 0
    
    # perform linear interpolations to find the root
    x1 = a
    x2 = b
    x3 = 10 # arbitrary high number
    i = 1
    while abs(func(x3)) > tol and i <= maxiter:
        # interpolate
        f1 = func(x1)
        f2 = func(x2)
        x3 = (f1*x2 - f2*x1) / (f1 - f2)
        f3 = func(x3)
        
        # prepare for next interpolation
        x1 = x2
        x2 = x3
        i += 1
    
    if i > maxiter:
        print('Maximum number of iterations has been reached.')
    return x3, f3, i    

root, yval, i = regulaFalsi(0.5, 1.0)
print('The root is located at x = ' + str(root) + ' after ' + str(i) + ' number of iterations.')
print('This value should be close to zero:  ' + str(yval))
    