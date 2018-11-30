'''
Created on Nov 25, 2018

@author: liuyanan
'''
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

style.use('fivethirtyeight')

fig = plt.figure()

ax1 = fig.add_subplot(111, projection='3d')


x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([5, 6, 3, 4, 1, 2, 8, 9, 6, 6])
z = np.array([[1, 4, 2, 7, 9, 4, 5, 2, 5, 6]])

x2 = np.array([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10])
y2 = np.array([-5, -6, -3, -4, -1, -2, -8, -9, -6, -6])
z2 = np.array([[-1, -4, -2, -7, -9, -4, -5, -2, -5, -6]])


x3 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y3 = np.array([5, 6, 3, 4, 1, 2, 8, 9, 6, 6])
z3 = np.zeros(10)


def intro():

    # x, y, z = axes3d.get_test_data(0.05)
    ax1.plot_wireframe(x, y, z)

    ax1.set_xlabel('x axis')
    ax1.set_ylabel('y axis')
    ax1.set_zlabel('z axis')

    plt.show()


def scatter():

    ax1.scatter(x, y, z)

    ax1.scatter(x2, y2, z2, c='r', marker='x')

    ax1.set_xlabel('x axis')
    ax1.set_ylabel('y axis')
    ax1.set_zlabel('z axis')

    plt.show()


def barchart():
    dx = np.ones(10)
    dy = np.ones(10)
    dz = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    ax1.bar3d(x3, y3, z3, dx, dy, dz)

    ax1.set_xlabel('x axis')
    ax1.set_ylabel('y axis')
    ax1.set_zlabel('z axis')

    plt.show()


def conclusion():
    x, y, z = axes3d.get_test_data()

    ax1.plot_wireframe(x, y, z, rstride=2, cstride=2, linewidth=1)
    ax1.set_xlabel('x axis')
    ax1.set_ylabel('y axis')
    ax1.set_zlabel('z axis')

    plt.show()


if __name__ == '__main__':
    # intro()
    # scatter()
    # barchart()
    conclusion()
