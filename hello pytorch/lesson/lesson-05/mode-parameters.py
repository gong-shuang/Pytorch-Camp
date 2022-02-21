import torch
import torch.nn as nn

# 在torch中，只会处理2维的数据
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
# x.pow(2)的意思是x的平方
y = x.pow(2) + 0.2 * torch.rand(x.size())


class Net(torch.nn.Module):  # 继承torch的module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()  # 继承__init__功能
        # 定义每一层用什么样的样式
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden,bias= 1)  # 隐藏层线性输出
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden,bias=1)  # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output,bias=1)  # 输出层线性输出

    def forward(self, x):
        # 激励函数（隐藏层的线性值）
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.predict(x)  # 输出值
        return x


net = Net(2, 5, 3)
print(net)
print(net.parameters())
# paras = list(net.parameters())
# for num,para in enumerate(paras):
#     print('number:',num)
#     print(para)
#     print('_____________________________')