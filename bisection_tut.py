'''
Introduction to Bisection Root-Finding in Python

Name: Kevin Trinh
Goal: Find the root of log(3x/2)
'''

import math
import scipy as sp

def func(x):
    '''The function that we are finding the root of.'''
    return sp.log(1.5 * x)

def bisection(x1, x2, tol=1e-15, maxiter=1000):
    '''Perform bisection algorithm. Uses in f(xnew) for tolerance criterion.'''
    # check that root is within bracket before starting bisection
    assert func(x1) * func(x2) < 0
    
    # update brackets to find the root
    i = 1
    xnew = (x1 + x2) / 2.0
    while abs(func(xnew)) > tol and i <= maxiter:
        f1 = func(x1)
        f2 = func(x2)
        xnew = (x1 + x2) / 2.0
        fnew = func(xnew)
        if (f1 * fnew > 0):
            x1 = xnew
            f1 = fnew
        else:
            x2 = xnew
            f2 = fnew
        i += 1
    if i > maxiter:
        print('Maximum number of iterations has been reached.')
    return xnew, fnew, i

root, yval, i = bisection(0.5, 1.0)
print('The root is located at x = ' + str(root) + ' after ' + str(i) + ' number of iterations.')
print('This value should be close to zero:  ' + str(yval))
        
    