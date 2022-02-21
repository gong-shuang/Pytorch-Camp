# -*- coding:utf-8 -*-
"""
@file name  : lesson-03-Linear-Regression.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2018-10-15
@brief      : 一元线性回归模型
"""
import torch
import matplotlib.pyplot as plt
torch.manual_seed(106)

lr = 0.01  # 学习率    20191015修改

# 创建训练数据
x = torch.rand(20, 1) * 10  # x data (tensor), shape=(20, 1)
y = 2*x + (5 + torch.randn(20, 1))  # y data (tensor), shape=(20, 1)

print(" x:{}   y:{}".format(x, y)) #每次生成的x，y都是一样的，这个因为最上面的那句，设置了随机种子，只要随机种子不变，x，y就不会变化。

# 构建线性回归参数
w = torch.randn((1), requires_grad=True)
b = torch.zeros((1), requires_grad=True)

print(" begin w:{}   b:{}".format(w, b))

# for iteration in range(1000):
#
#     # print(" {} w:{}   b:{}".format(iteration, w, b))
#
#     # 前向传播
#     wx = torch.mul(w, x)
#     y_pred = torch.add(wx, b)
#
#     # 计算 MSE loss
#     loss = (0.5 * (y - y_pred) ** 2).mean()
#
#     # 反向传播
#     loss.backward()
#
#     # 更新参数
#     b.data.sub_(lr * b.grad)
#     w.data.sub_(lr * w.grad)
#
#     # 清零张量的梯度   20191015增加
#     w.grad.zero_()
#     b.grad.zero_()
#
#     # 绘图
#     if iteration % 100 == 0:
#         plt.cla()   # 防止社区版可视化时模型重叠2020-12-15
#         plt.scatter(x.data.numpy(), y.data.numpy())
#         plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=5)
#         plt.text(2, 20, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
#         plt.xlim(1.5, 10)
#         plt.ylim(8, 28)
#         plt.title("Iteration: {}\nw: {} b: {}".format(iteration, w.data.numpy(), b.data.numpy()))
#         plt.pause(0.5)
#
#         if loss.data.numpy() < 1:
#             break
#     plt.show()

print(" end w:{}   b:{}".format(w, b))