'''
Demonstrating Euler's method

Name: Kevin Trinh
Goal: Approximate the solution to a differential equation.
'''
import matplotlib.pyplot as plt

def diffeq(x,y):
    '''The ordinary differential equation that we are trying to approximate.'''
    dydx = .1*x*y + .5*y
    return dydx

def eulersMethod(func, x_0, y_0, h, end):
    '''Use Euler's method on any given function. Stop approximating the solution
    past a given x-value, end.'''
    # for plotting purposes
    x_array = []
    y_array = []
    
    # prepare for loop
    steps = int((end - x_0) / h)
    x = x_0
    y = y_0
    
    # main loop
    while x <= end:
        y += func(x,y)*h
        x += h
        y_array.append(y)
        x_array.append(x)
    return x_array, y_array

def plot():
    '''Plot results of Euler's method with 5 different step sizes.'''
    # obtain data
    x1_array, y1_array = eulersMethod(diffeq, 0, 1, 1.0, 6)
    x2_array, y2_array = eulersMethod(diffeq, 0, 1, 0.5, 6)
    x3_array, y3_array = eulersMethod(diffeq, 0, 1, 0.25, 6)
    x4_array, y4_array = eulersMethod(diffeq, 0, 1, 0.01, 6)
    x5_array, y5_array = eulersMethod(diffeq, 0, 1, 0.001, 6)
    
    # plot data
    plt.plot(x1_array, y1_array, 'r-', label='h = 1.0')
    plt.plot(x2_array, y2_array, 'b-', label='h = 0.5')
    plt.plot(x3_array, y3_array, 'g-', label='h = 0.1')
    plt.plot(x4_array, y4_array, 'y-', label='h = 0.01')
    plt.plot(x5_array, y5_array, 'm-', label='h = 0.001')
    
    # set graphing window, labels, and legend
    ymax = max(y3_array)
    plt.axis([1, 6, 0, ymax])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    
plot()
    