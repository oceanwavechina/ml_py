'''
Created on Nov 25, 2018

@author: liuyanan
'''

import random
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import matplotlib.dates as mdatas
import matplotlib.ticker as mticker
from mpl_finance import candlestick_ohlc as candlestick_ohlc

style.use('fivethirtyeight')


fig = plt.figure(facecolor='#f0f0f0')


def create_plots():
    xs = []
    ys = []

    for i in range(10):
        x = i
        y = random.randrange(10)

        xs.append(x)
        ys.append(y)

    return xs, ys


def demo_1():
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    x, y = create_plots()
    ax1.plot(x, y)

    x, y = create_plots()
    ax2.plot(x, y)

    plt.show()


def demo_2():
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(212)

    x, y = create_plots()
    ax1.plot(x, y)

    x, y = create_plots()
    ax2.plot(x, y)

    x, y = create_plots()
    ax3.plot(x, y)

    plt.show()


def subplot_2grid():
    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=1, colspan=1)
    ax2 = plt.subplot2grid((6, 1), (1, 0), rowspan=4, colspan=1)
    ax3 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1)

    x, y = create_plots()
    ax1.plot(x, y)

    x, y = create_plots()
    ax2.plot(x, y)

    x, y = create_plots()
    ax3.plot(x, y)

    plt.show()


MA1 = 10
MA2 = 30


def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    smas = np.convolve(values, weights, 'valid')
    return smas


def high_minus_low(hights, lows):
    return hights - lows


highs = [11, 12, 15, 14, 13]
lows = [5, 6, 2, 6, 7]

h_1 = list(map(high_minus_low, highs, lows))
print(h_1)


def bytespdata2num(fmt):
    strconverter = mdatas.strpdate2num(fmt)
    return strconverter


def candlestick_with_subplot():
    # fig = plt.figure()
    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=1, colspan=1)
    plt.title('ICBC')
    plt.ylabel('H-L')

    ax2 = plt.subplot2grid((6, 1), (1, 0), rowspan=4, colspan=1, sharex=ax1)
    # after it's plot
    # plt.xlabel('Date')
    plt.ylabel('Price')
    ax2v = ax2.twinx()

    ax3 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)
    plt.ylabel('MAvgs')

    date, closep, highp, lowp, openp, volume = np.loadtxt('601398.csv',
                                                          delimiter=',',
                                                          skiprows=60,
                                                          unpack=True,
                                                          encoding='utf-8',
                                                          converters={0: bytespdata2num('%Y-%m-%d')},
                                                          usecols=(0, 3, 4, 5, 6, 13))

    x = 0
    y = len(date)
    ohlc = []

    while x < y:
        append_me = date[x], openp[x], highp[x], lowp[x], closep[x], volume[x]
        ohlc.append(append_me)
        x += 1

    ma1 = moving_average(closep, MA1)
    ma2 = moving_average(closep, MA2)
    start = len(date[MA2 - 1:])
    h_1 = list(map(high_minus_low, highp, lowp))

    ax1.plot_date(date[-start:], h_1[-start:], '-', label='H_L')
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3, prune='lower'))

    candlestick_ohlc(ax2, ohlc[-start:], width=0.8, colorup='r', colordown='g')

    for label in ax2.xaxis.get_ticklabels():
        label.set_rotation(45)
    ax2.xaxis.set_major_formatter(mdatas.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(10))
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='upper'))
    ax2.grid(True)

    # annotation with style
    bbox_props = dict(boxstyle='larrow', fc='w', ec='k', lw=1)
    ax2.annotate(str(closep[-1]), (date[-1], closep[-1]),
                 xytext=(date[-1] + 4, closep[-1]), bbox=bbox_props)

    ax2v.plot([], [], color='#0079a3', alpha=0.4, label='Volume')
    ax2v.fill_between(date[-start:], 0, volume[-start:], facecolor='#0079a3', alpha=0.4)
    ax2v.axes.yaxis.set_ticklabels([])
    ax2v.grid(False)
    ax2v.set_ylim(0, 10 * volume.max())

    print(len(date), len(ma1), len(ma2))
    ax3.plot(date[-start:], ma1[-start:], linewidth=1, label=(str(MA1) + 'MA'))
    ax3.plot(date[-start:], ma2[-start:], linewidth=1, label=(str(MA2) + 'MA'))
    ax3.fill_between(date[-start:], ma2[-start:], ma1[-start:], where=(ma1[-start:] < ma2[-start:]),
                     facecolor='r', edgecolor='r', alpha=0.5)

    ax3.fill_between(date[-start:], ma2[-start:], ma1[-start:], where=(ma1[-start:] > ma2[-start:]),
                     facecolor='g', edgecolor='g', alpha=0.5)

    ax3.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune='upper'))
    ax3.xaxis.set_major_formatter(mdatas.DateFormatter('%Y-%m-%d'))
    ax3.xaxis.set_major_locator(mticker.MaxNLocator(10))
    for label in ax3.xaxis.get_ticklabels():
        label.set_rotation(45)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # plt.legend()
    plt.subplots_adjust(left=0.11, bottom=0.24, right=0.90, top=0.90, wspace=0.9, hspace=0)

    ax1.legend()
    leg = ax1.legend(loc=9, ncol=2, prop={'size': 11})
    leg.get_frame().set_alpha(0.4)
    ax2v.legend()
    leg = ax2v.legend(loc=9, ncol=2, prop={'size': 11})
    leg.get_frame().set_alpha(0.4)
    ax3.legend()
    leg = ax3.legend(loc=9, ncol=1, prop={'size': 11})
    leg.get_frame().set_alpha(0.4)

    plt.show()
    fig.savefig('icbc.png', facecolor=fig.get_facecolor())


if __name__ == '__main__':
    # demo_1()
    # demo_2()
    # subplot_2grid()
    candlestick_with_subplot()
