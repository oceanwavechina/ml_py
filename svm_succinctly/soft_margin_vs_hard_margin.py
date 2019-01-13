import numpy as np

"""
    这个是比较 regularization 后的 svm 与之前的容错区别
"""

w = np.array([0.4, 1])
b = -10

x = np.array([6, 8])
y = -1

def constraint(w, b, x, y):
    return  y * (np.dot(w, x) + b)

def hard_constraint_is_satisfied(w, b, x, y):
    return constraint(w, b, x, y) >= 1

def soft_constraint_is_satisfied(w, b, x, y, zeta):
    return constraint(w, b, x, y) >= 1 - zeta

print('satisfy hard constriant? : ', hard_constraint_is_satisfied(w, b, x, y))

print('satisfy hard constriant? : ', soft_constraint_is_satisfied(w, b, x, y, zeta=2))

