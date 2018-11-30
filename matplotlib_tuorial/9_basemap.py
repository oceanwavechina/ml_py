'''
Created on Nov 25, 2018

@author: liuyanan
'''
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


def demo_1():
    m = Basemap(projection='mill',
                llcrnrlat=25,
                llcrnrlon=-130,
                urcrnrlat=50,
                urcrnrlon=-60,
                resolution='1')

    m.drawcoastlines()
    m.drawcountries(linewidth=2)

    plt.title('Basemap Tutorial')
    plt.show()


def draw():

    xs = []
    ys = []

    m = Basemap(projection='mill',
                llcrnrlat=25,
                llcrnrlon=-130,
                urcrnrlat=50,
                urcrnrlon=-60,
                resolution='1')

    m.drawcoastlines()
    m.drawcountries(linewidth=2)
    m.drawstates(color='b')

    NYClat, NYClon = 40.7127, -74.0059
    xpt, ypt = m(NYClon, NYClat)
    xs.append(xpt)
    ys.append(ypt)
    m.plot(xpt, ypt, 'c*', markersize=15)

    LAlat, LAlon = 34.05, -118.25
    xpt, ypt = m(LAlon, LAlat)
    xs.append(xpt)
    ys.append(ypt)
    m.plot(xpt, ypt, 'g^', markersize=15)

    m.plot(xs, ys, color='r', linewidth=3, label='flight')
    m.drawgreatcircle(NYClon, NYClat, LAlon, LAlat, color='c',
                      label='Arc')

    plt.legend(loc=4)

    plt.show()
