import numpy as np
import matplotlib.pyplot as plt
from 深度学习.common.functions import *
from 深度学习.common.gradient import numerical_gradient

# 定义梯度下降法的函数
def gradient_descent(f, init_x, lr=0.01, num_iter=100):
    x = init_x
    #定义列表保存x的变化
    x_history = []
    # 迭代num_iter次
    for i in range(num_iter):
        x_history.append(x.copy())
        # 计算梯度
        grad = numerical_gradient(f, x)
        # 梯度下降 更新参数
        x -= lr * grad
    return x, np.array(x_history)

# 定义目标函数 f(x1,x1) = x1^2 + x2^2
def f(x):
    return x[0]**2 + x[1]**2

#主流程
if __name__ == '__main__':
    # 1.初始参数
    init_x = np.array([-3.0, 4.0])
    # 2.定义超参数
    lr = 0.1
    num_iter = 20
    # 3. 梯度下降
    x, x_history = gradient_descent(f, init_x, lr, num_iter)
    print("最小值店为",x)

    # 画图
    plt.plot( [-5, 5], [0, 0], '--b')
    plt.plot( [0, 0], [-5, 5], '--b')
    plt.scatter(x_history[:,0], x_history[:,1], c='r')
    plt.xlim(-3.5, 3.5)
    plt.ylim(-4.5, 4.5)
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.show()