import numpy as np
from 深度学习.common.functions import *
from 深度学习.common.gradient import numerical_gradient

# 定义一个简单神经网络类
class SimpleNet:
    #初始化
    def __init__(self):
        self.W = np.random.randn(2,3)
    # 前向传播
    def forward(self, x):
        a = x @ self.W
        y = softmax(a)
        return y
    # 计算损失
    def loss(self, x, t):
        y = self.forward(x)
        return cross_entropy(y, t)


# 主流程
if __name__ == '__main__':
    # 1. 定义数据
    x = np.array([0.6,0.9])
    t = np.array([0,0,1])
    # 2. 初始化网络
    network = SimpleNet()
    print("初始权重：", network.W)
    # 3. 前向传播 计算梯度
    f = lambda w : network.loss(x, t)
    gradw = numerical_gradient(f,network.W)
    print("梯度：", gradw)


