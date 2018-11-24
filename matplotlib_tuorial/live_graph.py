'''
Created on Nov 24, 2018

@author: liuyanan
'''

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('fivethirtyeight')


fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)


def animate(i):
    graph_data = open('example.csv', 'r').read()
    lines = graph_data.split('\n')
    xs = []
    ys = []

    for line in lines:
        if len(line) > 1:
            x, y = line.split(',')
            xs.append(int(x))
            ys.append(int(y))

    ax1.clear()
    ax1.plot(xs, ys)


def live():
    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()


if __name__ == '__main__':
    live()
