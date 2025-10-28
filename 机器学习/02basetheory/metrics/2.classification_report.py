import numpy as np
from sklearn.datasets import make_classification #自动生成分类数据集的函数
from sklearn.model_selection import train_test_split #划分数据集
from sklearn.metrics import classification_report #生成分类评估报告
from sklearn.linear_model import LogisticRegression #逻辑回归模型

# 生成数据集
X,y = make_classification(n_samples=1000, n_features=20, n_classes=2,random_state=42)
# print(X.shape)
# print(y.shape)

# 2. 划分数据集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# 3. 定义模型
model = LogisticRegression()

# 4. 训练模型
model.fit(X_train,y_train)

# 5. 预测结果
y_pred = model.predict(X_test)

# 6. 生成评估报告
report = classification_report(y_test,y_pred)
print(report)


# 获取预测正类的概率值
y_pred_proba = model.predict_proba(X_test)[:,1]
# print(y_pred_proba)
# 计算AUC值
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test,y_pred_proba)
print(auc)