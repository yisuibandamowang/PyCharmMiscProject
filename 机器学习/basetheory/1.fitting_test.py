import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from jupyter_server.transutils import trans
from sklearn.model_selection import train_test_split #划分许训练集和测试集
from sklearn.linear_model import LinearRegression #线性回归模型
from sklearn.preprocessing import PolynomialFeatures #构建多项式特征
from sklearn.metrics import mean_squared_error #均方误差


'''
    1. 生成数据
    2. 划分测试机和训练集
    3. 定义模型（线性回归模型）
    4. 训练模型
    5. 预测结果  计算损失
'''

#1. 生成数据
X = np.linspace(-3,3,300).reshape(-1,1)
Y = np.sin(X) + np.random.uniform(low=-0.5,high=0.5,size=(300,1))
print(X.shape)
print(Y.shape)

# 画出散点图
fig,ax = plt.subplots(1,3,figsize=(15,4))
ax[0].scatter(X,Y,c='y')
ax[1].scatter(X,Y,c='y')
ax[2].scatter(X,Y,c='y')
# plt.show()

#2. 划分测试机和训练集
trainX,testX,trainY,testY = train_test_split(X,Y,test_size=0.2,random_state=42)

#3. 定义模型（线性回归模型）
model = LinearRegression()

#划分 欠拟合
x_train1 = trainX
x_test1 = testX

#过拟合
#洽拟合

#4. 训练模型
model.fit(x_train1,trainY)

print(model.coef_)
print(model.intercept_)

#5. 预测结果  计算损失
y_pred1 = model.predict(x_test1)
test_loss1 = mean_squared_error(testY,y_pred1)
train_loss1 = mean_squared_error(trainY,model.predict(x_train1))

#画出拟合曲线 并写出测试误差和训练误差
ax[0].plot(X,model.predict(X),'r')
ax[0].text(-3,1.3,f"训练误差:{train_loss1:.2f}")
ax[0].text(-3,1,f"测试误差:{test_loss1:.2f}")

plt.rcParams['font.sans-serif'] = ['STHeiTi']
plt.rcParams['axes.unicode_minus'] = False
plt.show()