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


if __name__ == '__main__':
    # part1()
    part2()
