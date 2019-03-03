
'''
数据源 
    https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls

把非数值的字段转换成一个集合中的id

这个是对titantic乘客的分类，两个问题
    1. 结果不稳定的情况怎么处理
    2. 现实中怎么判断数据集是可分类的，或是有意义的分类
        像titantic乘客的分类，感觉意义不大
        因为维度比较多，所以不好plot一个分布来直观的看一下，如果维度少，可以先plot一下
'''

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing, model_selection
import pandas as pd

style.use('ggplot')

df = pd.read_excel('titanic.xls')
df.drop(['body', 'name'], 'columns', inplace=True)
df.convert_objects(convert_numeric=True)
# df.apply(pd.to_numeric, errors='ignore')
df.fillna(0, inplace=True)
# print(df.head())

def handle_non_numerical_data(df):
    '''
        把某一列中所有出现的的非数字类型的字符，组合成一个集合，然后对没给元素编号(这个编号每次跑都是不一样的)
        这个数值编号就是我要的结果，即相当于转成了数值表示
    '''
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

df = handle_non_numerical_data(df)


# 尝试提高正确率: 进行数据处理(不过结果还是不稳定)
# df.drop(['ticket'], 1, inplace=True)
df.drop(['boat'], 1, inplace=True)

X = np.array(df.drop(['survived'], 1)).astype(float)

# 尝试提高正确率: scale之后可以
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))

