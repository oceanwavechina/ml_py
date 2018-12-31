'''
Created on Dec 30, 2018

@author: liuyanan
'''
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from dataset import get_dataset, linearly_seperable as ls

style.use('ggplot')

counter = 0


def plot_hyperline(w, is_final=False):
    # 画出当前的hyperplane
    '''
        关于y的计算:
            假设二维平面上我们给定hyperplane 为 w=(0.4, 1.0), b=-9
            则hyperplane上的点满足 xw+b=0, 即 0.4 * X1 +1.0 * X2 - 9 = 0
            转化成 augmented_X 后有 X=(1, x1, x2), w=(w0, w1, w2), 即 X.w=0
            有 1 * w0 + x1 * w1 + x2 * w2 = 0
            所以 x2 = (1 * w0 + x1 * w1) / w2 * -1
    '''
    global counter
    counter += 1
    if counter % 20 == 0 or is_final:
        line_xs = np.array(range(0, 10))
        # line1_ys = [(w[0] + w[1] * xs) / w[2] * -1 for xs in line_xs]
        line1_ys = [(w[0] + w[1] * xs) / w[2] * -1 for xs in line_xs]
        plt.plot(line_xs, line1_ys, label='final' if is_final else str(counter))
        plt.ylim((0, 12))
        plt.legend()


def perception_learning_algo(X, y):
    # 因为我们的特征是3个(算上我们新加的为1的那一列)
    w = np.random.rand(3)
    # 获取这次模型预测错的数据
    misclassified_examples = predict(hypothesis, X, y, w)

    # 如果有预测错的数据就要继续迭代
    while misclassified_examples.any():
        x, expected_y = pick_one_from(misclassified_examples, X, y)
        plot_hyperline(w)
        '''
            这是感知器算法的核心思想：
            如果expected_y 是1， 则我们预测的是-1， 所以我们预测模型中预设的 w和w 向量的夹角比预期的大了，
            我们就要缩小这个夹角，缩小夹角要让两个向量相加, 故 w=w+x
        '''
        w = w + x * expected_y

        # 接下来就是用更新有后的w重新预测
        misclassified_examples = predict(hypothesis, X, y, w)

    plot_hyperline(w, True)
    return w


def hypothesis(x, w):
    '假设， 返回的是个函数'
    h = np.sign(np.dot(x, w))
    return h


def predict(hypothesis_func, X, y, w):
    '根据hypothesis计算数据集的分类，并返回分错的数据'
    predictions = np.apply_along_axis(hypothesis_func, 1, X, w)
    misclassified = X[y != predictions]
    return misclassified


def pick_one_from(misclassified_examples, X, y):
    '这个啥意思'
    np.random.shuffle(misclassified_examples)
    x = misclassified_examples[0]
    # print('x:', x)
    # print('X:', X)
    index = np.where(np.all(X == x, axis=1))
    # print('np.all(X == x, axis=1): ', np.all(X == x, axis=1))
    # print('np.where(np.all(X == x, axis=1)):', np.where(np.all(X == x, axis=1)))
    return x, y[index]


def test():
    np.random.seed(88)

    X, y = get_dataset(ls.get_training_examples)
    X_augment = np.c_[np.ones(X.shape[0]), X]
    # 这里加一列，是为y=ax+b中的b的参数，相当于(1,a).(b,x), 其中的
    print('X_augment:', X_augment, 'y:', y)
    w = perception_learning_algo(X_augment, y)

    # x, y, z = axes3d.get_test_data(0.05)
    plt.plot(np.reshape(X[:7, 0], (-1, )), np.reshape(X[:7, 1], (-1, )), '^')
    plt.plot(np.reshape(X[7:, 0], (-1, )), np.reshape(X[7:, 1], (-1, )), 'o')

    plt.show()


if __name__ == '__main__':
    test()
