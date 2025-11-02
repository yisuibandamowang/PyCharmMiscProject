from 深度学习.common.functions import *
# Relu
class Relu:
    # 初始化
    def __init__(self):
        # 内部属性 记录那些 x<=0
        self.mask = None
    # 前向传播
    def forward(self, x):
        self.mask = (x <= 0)
        y = x.copy()
        # 将 x<=0 的值改为0
        y[self.mask] = 0
        return y
    # 反向传播
    def backward(self, dy):
        dx = dy.copy()
        # 将 x<=0 的值改为0
        dx[self.mask] = 0
        return

# Sigmoid
class sigmoid:
    #初始化
    def __init__(self):
        #定义一个属性 记录输出值y 用于反向传播时计算梯度
        self.y = None
    # 前向传播
    def forward(self, x):
        y = sigmoid(x)
        self.y = y
        return y
    # 反向传播
    def backward(self, dy):
        dx = dy * self.y * (1.0 - self.y)
        return dx