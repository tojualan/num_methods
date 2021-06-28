import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np

def banach(fun,x0,err = 1e-5):
    """
    Root finder using Banach fixed point theorem
    """
    delta = 1
    x = x0
    while delta > err:
        x = fun(x)
        delta = np.abs((x-fun(x)))
        print(x,delta)

    return x

def newton(fun,dfun,x0, err = 1e-5):
    """
    Newton's method for root finding
    """
    delta  = 1
    x = x0
    # while delta > err:
    for i in range(10):
        x = x - fun(x)/dfun(x)
        delta = np.abs(x-fun(x))

    return x

def euler(ode,ic,xrange,steps=100):
    """
    Euler's method for ODEs
    """

    #assume ic = y(a), with xrange = [a,b]
    h = (xrange[1]-xrange[0])/steps

    x = [xrange[0]]
    y = [ic]
    for i in range(1,steps+1):
        x.append(x[i-1]+h)
        y.append(y[i-1]+h*ode(x[i-1],y[i-1]))

    return x, y


def implicit_euler(ode,ic,xrange,step_size,theta):
    """
    Implicit Euler for ODEs
    """
    #assume ic = y(a), with xrange = [a,b]
    h = step_size
    n = int((xrange[1]-xrange[0])/h)

    x = [xrange[0]]
    y = [ic]
    for i in range(1,n+1):
        x.append(x[i-1]+h)
        ytmp = banach(lambda t: y[i-1]+h*((1-theta)*ode(x[i-1],y[i-1])
                           +theta*ode(x[i],t)),y[i-1])
        y.append(ytmp)

    return x, y



def RK4(ode, ic, xrange, h):
    """
    Fourth order Runge-Kutta algoritm for ODEs
    """
    # assume ic = y(a), with xrange = [a,b]
    n = int((xrange[1]-xrange[0])/h)
    m = len(ic)

    x = xrange[0]
    y = ic


    xsol = np.empty(0)
    xsol = np.append(xsol,x)

    ysol = np.array(y)

    for i in range(n):
        k1 = ode(x,y)
        k2 = ode(x+0.5*h,y + k1*0.5*h)
        k3 = ode(x+0.5*h,y + 0.5*h*k2)
        k4 = ode(x+h,y + h*k3)
        for j in range(m):
            y[j] += 1/6 * h * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j])

        x += h
        xsol = np.append(xsol, x)
        ysol = np.vstack((ysol, y))

    return xsol, ysol
