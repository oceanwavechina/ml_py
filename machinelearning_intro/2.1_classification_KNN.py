'''
Created on Dec 16, 2018

@author: liuyanan

NOTE:
    基本思想：
        对于一个要预测的 x 来说，在样本空间中选取 K 个最相近的样本，x属于这个K个样本中的大多数和x接近的那个分类
    对于一个可以进行linear regression的数据集，也可以用到两条line的距离进行分类，但是non-linear的数据集是不使用线性回归的方法的
    但是对于non-linear的数据集，knn依然是适用的
'''

import numpy as np
from sklearn import neighbors, model_selection
import pandas as pd


def knn_test():
    df = pd.read_csv('breast-cancer-wisconsin.data')
    # 处理不完整的数据, 不是直接drop掉
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)  # id这一列没什么用

    X = np.array(df.drop(['class'], 1))
    y = np.array(df['class'])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)

    # 这个分数叫accuracy, 区别于confidence
    accuracy = clf.score(X_test, y_test)
    print(accuracy)

    example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [10, 10, 10, 10, 10, 10, 10, 10, 10]])
    example_measures = example_measures.reshape(len(example_measures), -1)
    print(example_measures)
    prediction = clf.predict(example_measures)
    print(prediction)
    return accuracy


if __name__ == '__main__':
    accuracies = []
    for i in range(25):
        accuracies.append(knn_test())

    print(sum(accuracies) / len(accuracies))
