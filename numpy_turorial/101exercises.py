'''
Created on Dec 3, 2018

@author: liuyanan

@from: https://www.machinelearningplus.com/python/101-numpy-exercises-python/
'''


import numpy as np


def part1():
    print(np.__version__)

    # 3. 创建一个boolean类型的数组
    print(np.full((3, 4), fill_value=True, dtype=bool))
    print(np.ones((3, 4), dtype=bool))
    print(np.zeros((3, 4), dtype=bool))

    ''' 元素查找 和 替换 '''
    # 4. 找到数组中所有符合条件的元素
    arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(arr % 2 == 1)
    print(arr[arr % 2 == 1])  # 这个不会修改原有的数组的绑定
    print(arr)

    # 5. 替换数组中符合条件的元素，直接修改原数组
    arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    arr[arr % 2 == 1] = -1
    print(arr)

    # 6. 替换数组中符合条件的元素，*不能* 修改原数组
    arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    out = np.where(arr % 2 == 1, -1, arr)
    print(out)

    ''' 数组的变换 '''
    # 7. reshape一个数组
    arr = np.arange(10)
    newshape_arr = arr.reshape(2, -1)   # -1指的是系统计算这个值
    newshape_arr = arr.reshape(-1, 2)   # -1指的是系统计算这个值
    print(newshape_arr)

    # NOTE: 下边的合并，一定要是shape相同的数组才行
    a = np.arange(10).reshape(2, -1)
    b = np.repeat(1, 10).reshape(2, -1)
    # 8. 数组的纵向合并
    print(np.concatenate([a, b], axis=0))
    print(np.vstack((a, b)))
    print(np.r_[a, b])  # 注意这个是[], 不是()

    # 9. 数组的横向合并
    print(np.concatenate([a, b], axis=1))
    print(np.hstack((a, b)))
    print(np.c_[a, b])  # 注意这个是[], 不是()

    # 10. 生成自定义数组
    a = np.array([1, 2, 3])
    print(np.r_[np.repeat(a, 3), np.tile(a, 3)])  # 这个主要是区别 repeat和tile的不同

    # 11. 获取数组中的公共元素, 交集
    a = np.array([1, 2, 3, 2, 3, 4, 3, 4, 5, 6])
    b = np.array([7, 2, 10, 2, 7, 4, 9, 4, 9, 8])
    print(np.intersect1d(a, b, assume_unique=False, return_indices=True))  # 可以处理排序和不排序两种
    print(np.intersect1d(a, b, False))

    # 12. 移除另一个数组中出现的元素, 差集
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([5, 6, 7, 8, 9, 4, 4])
    print(np.setdiff1d(a, b, False))   # 注意顺序， 从a里边移除所有b中的元素

    # 13. 获取两个数组中元素相等的位置, 注意两个数组中的元素个数要相同
    a = np.array([1, 2, 3, 2, 3, 4, 3, 4, 5, 6])
    b = np.array([7, 2, 10, 2, 7, 4, 9, 4, 9, 8])
    print(np.where(a == b))

    # 14. 在数组中查找符合某一范围的元素
    a = np.array([2, 6, 1, 9, 10, 3, 27])
    index = np.where((a > 5) & (a <= 10))
    print(a[index])
    index = np.where(np.logical_and(a >= 5, a <= 10))
    print(a[index])
    print(a[(a >= 5) & (a <= 10)])

    # 15. 把对元素操作的函数，应用于向量，类似于python中的map
    a = np.array([6.000000009, 7, 9, 8, 6, 4, 5])
    b = np.array([6.00000001, 3, 4, 8, 9, 7, 1])

    def maxx(x, y):
        return x if x >= y else y

    pair_max = np.vectorize(maxx, otypes=[float])  # 可以指定类型
    print(pair_max(a, b))

    # 16. 17. 交换二维数组中的列 或 行
    # http://docs.scipy.org/doc/numpy-1.15.4/reference/arrays.indexing.html#arrays-indexing
    arr = np.arange(9).reshape(3, 3)
    print(arr[:, [1, 0, 2]])
    print(arr[[1, 0, 2], :])

    # 18. 19. 翻转二维数组中的行 或 列
    # http://docs.scipy.org/doc/numpy-1.15.4/reference/arrays.indexing.html#arrays-indexing
    # 对于一个数组index而言，有如下格式: element = array[rowstart:rowstop:rowstep, colstart:colstop:colstep]
    arr = np.arange(9).reshape(3, 3)
    print(arr[::-1])     # 所以这个是对行操作，step是-1
    print(arr[:, ::-1])   # 所以这个是对列操作，step是-1
    print(arr[:, ::1])   # step是1相当于没有变化

    # 取出数组中奇数位置的元素, 并对这些元素进行平方计算
    arr = np.arange(10)
    print(arr[1::2] ** 2)

    # 20. 创建一个二维数组， 其中元素范围是给定的，要求是float类型
    rand_arr = np.random.randint(low=5, high=10, size=(5, 3)) + np.random.random((5, 3))
    print(rand_arr)
    rand_arr = np.random.uniform(5, 10, size=(5, 3))
    print(rand_arr)


def part2():
    # 21. 22. 23. 24.  输出设置
    rand_arr = np.random.random([10, 8])
    np.set_printoptions(precision=30, threshold=1)
    print(rand_arr)

    # 26. 获取特定的列
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris = np.genfromtxt(url, delimiter=',', dtype=None, encoding='utf8')
    # print('25:', iris_1d, np.shape(iris_1d))
    species = np.array([row[4] for row in iris])
    print(species[:5])

    # 28. 计算数组的均值，中位数，标准差
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
    mu, med, sd = np.mean(sepallength), np.median(sepallength), np.std(sepallength)
    print(mu, med, sd)

    # 29. 把数组元素进行归一化
    S = (sepallength - sepallength.min()) / sepallength.ptp()
    print('29: ', S)

    # 36. 计算两列的相关性系数，当样本量较大时才有意义，是正相关还是负相关，以及相关程度
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0, 1, 2, 3])
    corr = np.corrcoef(iris[:, 0], iris[:, 2])[0, 1]
    print('36: ', corr)

    # 44. 二维数组中根据某一列进行排序, argsort()返回的是排序后的索引, 这个索引是全局的索引，所以可以用iris[index_array]的形式取数据
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris = np.genfromtxt(url, delimiter=',', dtype='object')
    print('44: ', iris[iris[:, 2].argsort()][:20])

    # 45. 找出出现次数最多的元素
    vals, counts = np.unique(iris[:, 2], return_counts=True)
    # 这里的 vals 和 counts 中的元素是一一对应的，找到counts的最大值的那个位置，也就找到了vals最大值的位置了
    print(vals, counts)
    print('45: ', vals[np.argmax(counts)])

    # 48. 获取top n value 的位置和他们的值
    np.random.seed(100)
    a = np.random.uniform(1, 50, 20)
    print('48: ', a.argsort(), a[a.argsort()][:-5])

    # 58. 找到数组中重复的元素????
    np.random.seed(100)
    a = np.random.randint(0, 6, 8)
    np.set_printoptions(threshold=100)

    # 62. 计算两个数组的欧几里得距离，这个是简单的相似度计算
    a = np.array([1, 2, 3, 4, 5])
    b1 = np.array([4, 5, 6, 7, 8])
    b2 = np.array([3, 4, 5, 6, 7])
    print('dist(a, a): ', np.linalg.norm(a - a))
    print('dist(a, b1): ', np.linalg.norm(a - b1))
    print('dist(a, b2): ', np.linalg.norm(a - b2))
    # 归一化, 值越接近1， 相似度越高
    print('norm_dist(a, a): ', 1 / (1 + np.linalg.norm(a - a)))
    print('norm_dist(a, b1): ', 1 / (1 + np.linalg.norm(a - b1)))
    print('norm_dist(a, b2): ', 1 / (1 + np.linalg.norm(a - b2)))

    # 67. moving average
    # 以1、2、3、4、5共5个数为例，window为3，计算过程为：（1+2+3）/3=2，（2+3+4）/3=3，（3+4+5）/3=4。
    # 抚平短期波动， 反应长期趋势或周期
    Z = np.random.randint(10, size=10)
    print('moving average: ', np.convolve(Z, np.ones(3) / 3, mode='valid'))


if __name__ == '__main__':
    # part1()
    part2()
