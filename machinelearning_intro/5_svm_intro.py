'''
Created on Dec 18, 2018

@author: liuyanan

NOTE:
    我们就是要找到 the best sperate hyper-plane

'''
import numpy as np
from sklearn import model_selection, svm
import pandas as pd


def svm_test():
    df = pd.read_csv('breast-cancer-wisconsin.data')
    # 处理不完整的数据, 不是直接drop掉
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)  # id这一列没什么用, 如果不drop掉会影响模型的准确

    X = np.array(df.drop(['class'], 1))
    y = np.array(df['class'])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    clf = svm.SVC()
    clf.fit(X_train, y_train)

    # 这个分数叫accuracy, 区别于confidence
    accuracy = clf.score(X_test, y_test)
    print(accuracy)

    example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [10, 10, 10, 10, 10, 10, 10, 10, 10]])
    example_measures = example_measures.reshape(len(example_measures), -1)
    prediction = clf.predict(example_measures)
    print(prediction)
    return accuracy


if __name__ == '__main__':
    accuracies = []
    for i in range(1):
        accuracies.append(svm_test())

    print(sum(accuracies) / len(accuracies))
