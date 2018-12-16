'''
Created on Dec 16, 2018

@author: liuyanan

NOTE:
    欧几里得距离
        ρ = sqrt( (x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2 + ...)

    KNN 中 K的选取是个经验值
'''
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from collections import Counter
import pandas as pd
import random

style.use('fivethirtyeight')


def test_euclidean_distance():
    plot1 = [1, 3]
    plot2 = [2, 5]

    euclidean_distance = sqrt((plot1[0] - plot2[0])**2 + (plot1[1] - plot2[1]) ** 2)
    print(euclidean_distance)


def k_nearest_neghbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value  less than total voting groups!')

    # 在计算的时候用到了所有的数据
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])  # ˙这样就可以根据距离排序了

    # 选取出现最多的,前k个类别
    votes = [i[1] for i in sorted(distances)[:k]]
    # print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
    # print(vote_result, confidence)
    return vote_result, confidence


def toy_dataset():
    dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
    new_features = [5, 7]

    result = k_nearest_neghbors(dataset, new_features, 3)

    for i in dataset:
        for ii in dataset[i]:
            plt.scatter(ii[0], ii[1], s=100, color=i)
    plt.scatter(new_features[0], new_features[1], color=result)
    plt.show()


def real_dataset():
    df = pd.read_csv('breast-cancer-wisconsin.data')
    df.replace('?', -9999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    full_data = df.astype(float).values.tolist()
    # full_data = df.values.tolist() #如果不加astype(float), 有些值会是字符串类型的

    random.shuffle(full_data)
    test_size = 0.2
    test_size = 0.4
    train_set = {2: [], 4: []}
    test_set = {2: [], 4: []}
    train_data = full_data[:-int(test_size * len(full_data))]
    test_data = full_data[-int(test_size * len(full_data)):]

    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    # print(train_data)
    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neghbors(train_set, data, k=5)
            if group == vote:
                correct += 1
#             else:
#                 print(confidence)
            total += 1

    print('Accuracy:', correct / total)

    return correct / total


if __name__ == '__main__':

    accuracies = []
    for i in range(25):
        accuracies.append(real_dataset())

    print(sum(accuracies) / len(accuracies))
