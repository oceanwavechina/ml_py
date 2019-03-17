'''
    https://www.cnblogs.com/xfzhang/p/7261172.html
    
'''
import matplotlib.pylab as plt
from matplotlib import style
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import random

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

# centers = random.randrange(2, 8)
X, y = make_blobs(n_samples=50, centers=3, n_features=2)

# plt.scatter(X[:, 0], X[:, 1], s=150)
# plt.show()

colors = ["g", "r", "c", "b", "k", "m"]

class MeanShift:
    
    def __init__(self, radius=None, radius_norm_step=100): #4, 40
        self.radius = radius
        self.radius_norm_step = radius_norm_step

    def fit(self, data):

        # 计算 radius
        if self.radius == None:
            all_data_centroid = np.average(data, axis=0)
            all_data_norm = np.linalg.norm(all_data_centroid)
            self.radius = all_data_norm / self.radius_norm_step

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
                
                weights = [i for i in range(self.radius_norm_step)][::-1]

                # 对每个centroids都要遍历所有的数据集
                for featureset in data:
                    distance = np.linalg.norm(featureset-centroid)
                    if(distance == 0):
                        distance = 0.000000001
                    weight_index = int(distance / self.radius)
                    if weight_index > self.radius_norm_step-1:
                        weight_index = self.radius_norm_step-1
                    # to_add = (weights[weight_index]**2) * [featureset]
                    to_add = (weights[weight_index]*2) * [featureset]
                    in_bandwidth += to_add

                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))

            uniques = sorted(list(set(new_centroids)))

            to_pop =[]
            for i in uniques:
                for ii in uniques:
                    if i == ii:
                        pass
                    elif np.linalg.norm(np.array(i) - np.array(ii)) <= self.radius:
                        to_pop.append(ii)
                        break
            
            for i in to_pop:
                try:
                    uniques.remove(i)
                except:
                    pass


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

        self.classifications = {}
        for i in range(len(self.centroids)):
            self.classifications[i] = []
        for featureset in data:
            distance = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
            classification = distance.index(min(distance))
            self.classifications[classification].append(featureset)

    def predict(self, data):
        distance = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distance.index(min(distance))
        return classification

clf = MeanShift()
clf.fit(X)

centroids = clf.centroids
# plt.scatter(X[:, 0], X[:, 1], s=150)

for classification in clf.classifications:
    color = colors[classification]
    for featrueset in clf.classifications[classification]:
        plt.scatter(featrueset[0],featrueset[1], marker='x', color=color, s=150, linewidths=5)

for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150)

plt.show()

        
