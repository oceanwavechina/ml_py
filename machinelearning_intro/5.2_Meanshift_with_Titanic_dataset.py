'''
'''

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing, model_selection
import pandas as pd

style.use('ggplot')

df = pd.read_excel('titanic.xls')
original_df = pd.DataFrame.copy(df) # 拷贝
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

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df['cluster_group'] = np.nan

for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i]
n_clusters_ = len(np.unique(labels))

survival_rates = {}
for i in range(n_clusters_):
    temp_df = original_df[(original_df['cluster_group'] == float(i))]
    print(temp_df.describe())
    survival_cluster = temp_df[(temp_df['survived'] == 1)]
    survival_rate = len(survival_cluster) / len(temp_df)
    survival_rates[i] = survival_rate

print(survival_rates)
