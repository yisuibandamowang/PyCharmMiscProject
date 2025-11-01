import numpy as np
from 深度学习.common.functions import sigmoid, identity

# 初始化神经网络
def init_network():
    network = {}
    # 输入层和隐藏层的权重
    network["W1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network["b1"] = np.array([0.1, 0.2, 0.3])
    # 第二层参数
    network["W2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network["b2"] = np.array([0.1, 0.2])
    # 第三层参数
    network["W3"] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network["b3"] = np.array([0.1, 0.2])
    return network

#前向传播
def forward(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]
    # 逐层计算传递
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity(a3)
    return y

#测试主流程
network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
# 前向传播（预测）
print(y)