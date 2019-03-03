'''
    手动实现 K-means 算法, 步骤：
        1. 初始化 K 个 centroid
        2. 计算每个样本到centroid的距离
        3. 根据与最小的那个centorid的距离进行分类
        4. 对个group的元素求均值，然后更新centroid
        5. 判断有没有取到最优值/达到迭代次数
        6. goto 2
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
            [9, 11],])

plt.scatter(X[:, 0], X[:, 1], s=150)
plt.show()

colors = ["g", "r", "c", "b", "k", "m"]

class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}

        # 初始化centroids (可以randomly的方式), 有几个分类，就找几个centroid
        for i in range(self.k):
            self.centroids[i] = data[i]
        
        # 迭代
        for _ in range (self.max_iter):
            # 因为每次迭代时，样本的分类都有可能变
            self.classifications = {}

            # 对每个category创建一个list
            for i_cate in range(self.k):
                self.classifications[i_cate] = []
            
            # 计算每个样本到 “各个” centroid的距离
            for feature in data:
                # distances = [np.linalg.norm(feature-self.centroids[centroid]) for centroid in self.centroids]
                # 有几个K/categofy，distances里边就有几个元素
                distances = []
                for centroid in self.centroids:
                    # 计算每个样本和centroids的距离
                    distances.append(np.linalg.norm(feature-self.centroids[centroid]))

                # 选最小的那个的索引，这个索引就是分类
                classification = distances.index(min(distances))
                self.classifications[classification].append(feature)

            prev_centroids = dict(self.centroids)
            
            for classification in self.classifications:
                # pass
                # 注意这里的均值，axis=0， 是对每列求均值（也就是把所有行加起来除以行数）,最后列数是不变的
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
                print(np.average(self.classifications[classification], axis=0))
            
            # 判断是不是取到最优值了
            optimized = True
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        distances = []
        for centroid in self.centroids:
            # 计算每个样本和centroids的距离
            distances.append(np.linalg.norm(data-self.centroids[centroid]))

        # 选最小的那个的索引?
        classification = distances.index(min(distances))

        return classification

clf = K_Means()
clf.fit(X)

for centroid in clf.centroids:
    print(centroid)
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], 
                marker='o', color='k', s=150, linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for feature in clf.classifications[classification]:
        plt.scatter(feature[0], feature[1], marker='x', color=color, s=150, linewidths=5)

unkowns = np.array([[1, 3],
                    [8, 9],
                    [2, 2],
                    [10, 0],
                    [8, 2],])

for unkown in unkowns:
    classification = clf.predict(unkown)
    plt.scatter(unkown[0], unkown[1], marker='*', color=colors[classification], s=150, linewidths=5)

plt.show()