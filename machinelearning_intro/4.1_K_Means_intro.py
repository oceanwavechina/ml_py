'''
这个是非监督的学习算法

1. K-means (Flat Clustering), K是group的数量，期望分成多少个种类
'''

import matplotlib.pylab as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans

style.use('ggplot')


X = np.array([
            [1, 2],
            [1.5, 1.8],
            [5, 8],
            [8, 8],
            [1, 0.6],
            [9, 11],])

# plt.scatter(X[:, 0], X[:, 1], s=150)
# plt.show()

# clf = KMeans(n_clusters=2)
clf = KMeans(n_clusters=4)
clf.fit(X)

centroids = clf.cluster_centers_
labels = clf.labels_
print(labels)

colors = ["g", "r", "c", "b", "k", "m"]

for i in range(len(X)):
    # plt.plot(X[i][0], X[i][1], marker='o', color=colors[labels[i]], markersize=12)
    plt.plot(X[i][0], X[i][1], marker='o', color=colors[labels[i]], markersize=12)

plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, linewidths=5)
plt.show()