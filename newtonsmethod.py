'''
Newton's method in Python

Name: Kevin Trinh
Goal: Find the roots of an equation using Newton's method.
'''

import math

def func(x):
    '''Function that we are trying to solve.'''
    return 25 - x**2

def derFunc(x):
    '''Derivative of the function.'''
    return -2*x
    

def newton(f, df, x_0, tol=1e-15, maxiter=500):
    '''Perform Newton's method on any given function.'''
    i = 1
    x = x_0
    while (abs(f(x)) > tol and i <= maxiter):
        x -= f(x)/df(x)
        i += 1      
    return x, i

root1, i1 = newton(func, derFunc, 6)
root2, i2 = newton(func, derFunc, -4)
print('The solutions are ' + str(root1) + ' and ' + str(root2) + '.')
print('It took ' + str(i1) + ' and ' + str(i2) + ' iterations.')