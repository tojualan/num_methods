import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np
from numMethods import RK4


if __name__ == '__main__':

    def lorentz(t, y, sigma=10.,rho=28.,beta=8./3.):
        """
        Lorentz oscillator dy/dt = lorentz(t,y), y = [y0,y1,y2]
        """
        return np.array([sigma*(y[1]-y[0]),y[0]*(rho-y[2])-y[1],
                         y[0]*y[1]-beta*y[2]])

    x_sol, y_sol = RK4(lorentz, np.array([0.4, -0.7, 21.]),
                       np.array([0., 3.0]), 0.01)
    print(x_sol)
    print(y_sol)


    #Plot the result as an animation
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')


    def scatter_inst(i):
        """
        Scatter plot for instance i
        """
        try:
            ax.scatter(y_sol[i,0],y_sol[i,1],y_sol[i,2], c='royalblue', s=3)
        except IndexError:
            return 0

    ax.set_xlim(-20,20)
    ax.set_ylim(-30,30)
    ax.set_zlim(0,50)


    animator = ani.FuncAnimation(fig,scatter_inst,interval=10, frames = 300)
    plt.show()
