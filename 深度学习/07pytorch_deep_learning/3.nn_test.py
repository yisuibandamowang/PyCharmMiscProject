import torch
import torch.nn as nn
from datetime import datetime

# 自定义神经网络类
class Model(nn.Module):
    # 初始化
    def __init__(self, device='cpu'):
        super().__init__()
        # 定义三个线性层
        self.linear1 = nn.Linear(3, 4, device=device)
        nn.init.xavier_normal_(self.linear1.weight)
        self.linear2 = nn.Linear(4, 4, device=device)
        nn.init.kaiming_normal_(self.linear2.weight)
        self.out = nn.Linear(4, 2, device=device)
    # 前向传播
    def forward(self, x):
        x = self.linear1(x)
        x = torch.tanh(x)

        x = self.linear2(x)
        x = torch.relu(x)

        x = self.out(x)
        x = torch.softmax(x, dim=1)
        return x

# 统一定义全局变量：device
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"使用设备：{device}")

# 测试
# 1. 定义输入数据
x = torch.randn(10, 3, device=device)

# 2. 创建神经网络模型
model = Model(device=device)
model = model.to(device)
# 3. 前向传播
print(f"开始前向传播: {datetime.now()}")
output = model(x)
print(f"前向传播完成: {datetime.now()}")

print("神经网络输出为：", output)

# 查看参数
# 1. 逐个查看所有参数
print(model.linear1.weight)
print(model.linear1.bias)
print(model.linear2.weight)
print(model.linear2.bias)
print(model.out.weight)
print(model.out.bias)

print()

# 2. 调用parameters，直接查看所有参数
for param in model.parameters():
    print(param)

print()
for name, param in model.named_parameters():
    print(name, param)

# 3. 调用state_dict，得到所有参数的字典表示
print()
print(model.state_dict())

# 4. 查看模型的架构和参数数量
from torchsummary import summary
model.to('cpu')
summary(model, input_size=(3, ), batch_size=10, device='cpu')
model.to(device)