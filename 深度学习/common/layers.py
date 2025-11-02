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


# Affine 仿射层
class Affine:
    # 初始化
    def __init__(self, W, b):
        self.W = W
        self.b = b
        # 对输入数据 x 进行保存    方便方向传播计算梯度
        self.x = None
        self.orignal_x_shape = None
        # 对权重 dw 和偏执参数 db 进行保存
        self.dw = None
        self.db = None
    # 前向传播
    def forward(self, X):
        self.orignal_x_shape = X.shape
        self.X = X.reshape(X.shape[0], -1)
        y = np.dot(X, self.W) + self.b
        return y
    # 反向传播
    def backward(self, dy):
        dX = np.dot(dy, self.W.T)
        dX = dX.reshape(*self.orignal_x_shape)
        self.dW = np.dot(self.X.T, dy)
        self.db = np.sum(dy, axis=0)
        return dX