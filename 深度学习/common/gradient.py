import numpy as np
# 数值微分求导
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)


# 数值微分求梯度
def _numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    # 对每个元素求导
    for idx in range(x.size):
        tmp = x[idx]
        x[idx] = tmp + h
        fxh1 = f(x)
        x[idx] = tmp - h
        fxh2 = f(x)
        # 利用中心差分公式计算偏导
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        #恢复数组值
        x[idx] = tmp
    return grad

# 传入一个x是矩阵
def numerical_gradient(f, x):
    if x.ndim == 1:
        return _numerical_gradient(f, x)
    else :
        grad = np.zeros_like(x)
        # 对矩阵的每个元素求导
        for i,x in enumerate(x):
            grad[i] = _numerical_gradient(f, x)
        return grad
