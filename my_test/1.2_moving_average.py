import pandas as pd
import numbers as np
from datetime import datetime
import matplotlib.pyplot as plt
import pyEX as p

ticker = 'AMD'
timeframe = '1y'

df = p.chartDF(ticker, timeframe)
print(df.head())

df = df[['close']]
df.reset_index(level=0, inplace=True)
df.columns=['ds', 'y']
print(df.head())

def SMA():
    rolling_mean20 = df.y.rolling(window=20).mean()
    rolling_mean50 = df.y.rolling(window=50).mean()

    ax = plt.subplot(211)
    ax.plot(df.ds, df.y, label='AMD')
    ax.plot(df.ds, rolling_mean20, label='20 Day MA')
    ax.plot(df.ds, rolling_mean50, label='50 Day MA')
    ax.legend(loc='upper left')
    ax.set_title('SMA')
    
    

def EMA():
    ax = plt.subplot(212)
    exp20 = df.y.ewm(span=20, adjust=False).mean()
    exp50 = df.y.ewm(span=50, adjust=False).mean()
    ax.plot(df.ds, df.y, label='AMD')
    ax.plot(df.ds, exp20, label='20 Day EMA')
    ax.plot(df.ds, exp50, label='50 Day EMA')
    ax.legend(loc='upper left')
    ax.set_title('EMA')

SMA()
EMA()
plt.subplots_adjust()
plt.show()



