import numpy as np


# 随机梯度下降 SGD
class SGDR:
    # 初始化
    def __init__(self, lr=0.01):
        self.lr = lr

    # 参数更新 .  传入参数字典和梯度字典
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


# 动量法
class Momentum:
    # 初始化
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None  # 历史负梯度的加权和

    # 参数更新
    def update(self, params, grads):
        # 初始化 v
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        # 按照公式进行参数更新
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


# AdaGrad
class AdaGrad:
    # 构造函数
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    # 参数更新
    def update(self, params, grads):
        # 初始化 h
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        # 按照公式进行参数更新
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-8)

# RMSProp
class RMSProp:
    # 初始化
    def __init__(self, lr=0.01, decay=0.99):
        self.lr = lr
        self.decay = decay
        self.h = None
    # 更新方法
    def update(self, params, grads):
        # 对h进行初始化
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        # 按照公式进行参数更新
        for key in params.keys():
            self.h[key] *= self.decay
            self.h[key] += (1 - self.decay) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-8)