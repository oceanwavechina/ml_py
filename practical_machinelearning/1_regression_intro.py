'''
Created on Dec 5, 2018

@author: liuyanan

@desc:
'''

import pandas as pd
import quandl
import math

api_key = open('quandlapikey.txt', 'r').read().strip('\n')
df = quandl.get('WIKI/GOOGL', authtoken=api_key)

# volume指的是成交量
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# 最高价到最低价的变化
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100

# 收盘价到开盘价的变化
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

# 每一列就是一个feature
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forcast_col = 'Adj. Close'
# 现实的世界中，删掉nan的数据会损失数据，所以尽量不要用dropna之类的方法
df.fillna(-99999, inplace=True)

# 这个意思是根据之前1/100的数据来预测
forcast_out = int(math.ceil(0.01 * len(df)))
# 这样 label中就是1/100后数据的'Adj. Close'和当天的 'Adj. Volume' 在一行了，
# shift是负数是是往上移动, 下边的NAN补齐
df['label'] = df[forcast_col].shift(-forcast_out)
df.dropna(inplace=True)
print(df.head())
