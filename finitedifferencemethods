'''
Demonstrating Finite-Differencing Methods

Name: Kevin Trinh
Goal: Find numerical solutions and demonstrate stability
'''
import math
import matplotlib.pyplot as plt


### Global constants ###
x_0 = 0
f_0 = 100
k = 50

def f(x):
    '''Analytical solution to the diffeq: df/dx = -k*f'''
    return f_0*math.exp(-k*x)

def analytical(func,x_end):
    '''Create arrays to plot analytical solution.'''
    xarr=[]
    farr=[]
    x=0
    while x < x_end:
        farr.append(func(x))
        xarr.append(x)
        x+=.001
    return xarr, farr

def forward(h,x_end):
    '''Create arrays to plot numerical solution using
    forward-differencing (explicit scheme).'''
    # plotting arrays with starting values
    xarr=[x_0]
    farr=[f_0]
    x=x_0
    # obtain points using forward differencing
    while x <= x_end:
        fnext=(1-h*k)*farr[-1]
        x+=h
        xarr.append(x)
        farr.append(fnext)
    return xarr, farr

def backward(h,x_end):
    '''Create arrays to plot numerical solution using
    backward-differencing (implicit scheme).'''
    # plotting arrays with starting values
    xarr=[x_0]
    farr=[f_0]
    x=x_0
    # obtain points using backward differencing
    while x <= x_end:
        fnext=farr[-1]/(1+h*k)
        x+=h
        xarr.append(x)
        farr.append(fnext)
    return xarr, farr

def fplot():
    '''Plot results of Euler's method with 5 different step sizes.'''
    # obtain data
    x_array, f_array = analytical(f,.2)
    x1_array, f1_array = forward(.010,.15)
    x2_array, f2_array = forward(.020,.15)
    x3_array, f3_array = forward(.030,.15)
    x4_array, f4_array = forward(.040,.15)
    x5_array, f5_array = forward(.045,.15)
    # plot data
    plt.plot(x_array, f_array, 'k-', label='analytical')
    plt.plot(x1_array, f1_array, 'b-', label='h = .010')
    plt.plot(x2_array, f2_array, 'g-', label='h = .020')
    plt.plot(x3_array, f3_array, 'c-', label='h = .030')
    plt.plot(x4_array, f4_array, 'y-', label='h = .040')
    plt.plot(x5_array, f5_array, 'r-', label='h = .045')
    # set graphing window, labels, and legend
    plt.title('Stability for explicit scheme')
    plt.xlabel('x')
    plt.ylabel('f')
    plt.legend()
    plt.show()
    
def bplot():
    '''Plot results of Euler's method with 5 different step sizes.'''
    # obtain data
    x_array, f_array = analytical(f,.2)
    x1_array, f1_array = backward(.010,.15)
    x2_array, f2_array = backward(.020,.15)
    x3_array, f3_array = backward(.030,.15)
    x4_array, f4_array = backward(.040,.15)
    x5_array, f5_array = backward(.050,.15)
    # plot data
    plt.plot(x_array, f_array, 'k-', label='analytical')
    plt.plot(x1_array, f1_array, 'b-', label='h = .010')
    plt.plot(x2_array, f2_array, 'g-', label='h = .020')
    plt.plot(x3_array, f3_array, 'c-', label='h = .040')
    plt.plot(x4_array, f4_array, 'y-', label='h = .100')
    plt.plot(x5_array, f5_array, 'r-', label='h = 1.00')
    # set graphing window, labels, and legend
    plt.title('Stability for implicit scheme')
    plt.xlabel('x')
    plt.ylabel('f')
    plt.legend()
    plt.show()    
    
fplot()
bplot()
