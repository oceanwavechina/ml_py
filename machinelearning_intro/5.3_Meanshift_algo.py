'''
    https://www.cnblogs.com/xfzhang/p/7261172.html
    
'''
import matplotlib.pylab as plt
from matplotlib import style
import numpy as np

style.use('ggplot')


X = np.array([
            [1, 2],
            [1.5, 1.8],
            [5, 8],
            [8, 8],
            [1, 0.6],
            [9, 11],
            [8, 2],
            [10, 2],
            [9, 3],
            ])

# plt.scatter(X[:, 0], X[:, 1], s=150)
# plt.show()

colors = ["g", "r", "c", "b", "k", "m"]

class MeanShift:
    def __init__(self, radius=1): #4, 40
        self.radius = radius
    def fit(self, data):
        centroids = {}

        # 注意这个算法的初始值有n个centroids，区别于K-Means的初始值
        for i in range(len(data)):
            centroids[i] = data[i]
        
        while True:
            new_centroids = []
            
            # 这个for循环是把在radius内的元素放到对应的分类里边
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                
                # 对每个centroids都要遍历所有的数据集
                for featureset in data:
                    if np.linalg.norm(featureset-centroid) < self.radius:
                        in_bandwidth.append(featureset)
                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))

            uniques = sorted(list(set(new_centroids)))

            prev_centroids = dict(centroids)
            print(prev_centroids)

            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])
            
            optimized = True
            
            # 检查有没有收敛
            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                    break

            if optimized:
                break

        self.centroids = centroids

clf = MeanShift()
clf.fit(X)

centroids = clf.centroids
plt.scatter(X[:, 0], X[:, 1], s=150)

for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150)

plt.show()

        
