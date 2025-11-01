import numpy as np
import matplotlib.pyplot as plt
from 深度学习.common.gradient import numerical_diff

# 原函数 y = 0.01 * x^2 + 0.1 * x
def f(x):
    return 0.01 * x ** 2 + 0.1 * x

#切线方程函数,返回切线对应的函数
def tangent_line(f, x):
    y = f(x)
    # 计算x处切线的斜率(利用数值微分计算x处的导数)
    a = numerical_diff(f, x)
    print("斜率：", a)
    # 根据切线过（x，y）和斜率a，求出切线的方程
    b = y - a * x
    return lambda t: a * t + b

#定义画图范围
x = np.arange(0.0, 20.0, 0.1)
y = f(x)

#计算行x=5处的切线方程
f_line = tangent_line(f, x=5)
y_line = f_line(x)
#画出原函数曲线
plt.plot(x, y)
#画出切线
plt.plot(x, y_line)
plt.show()
