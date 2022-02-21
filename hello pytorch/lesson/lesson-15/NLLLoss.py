
#https://blog.csdn.net/Jeremy_lf/article/details/102725285

# nn.NLLLoss
# 此时，nn.NLLLoss的结果就是把上面的输出与Label对应的那个值拿出来，再去掉负号，再求均值。


import torch
from tools.common_tools import  set_seed

set_seed(1)  # 设置随机种子

# input=torch.randn(3,3)
input = torch.tensor([[1, 1, 1], [1, 2, 1], [1, 1, 1]], dtype=torch.float)

print('input:\n',input)

soft_input = torch.nn.Softmax(dim=0)
softmax_output=soft_input(input)
print('Softmax_output:\n',softmax_output)

#对softmax结果取log
log_output = torch.log(softmax_output)
print('log_output:\n',log_output)

# log_output:
#  tensor([[-0.7292, -0.9589, -0.7574],
#         [-0.7692, -1.6778, -0.9852],
#         [-2.9133, -0.8442, -1.8467]])

# 假设标签是[0,1,2]，第一行取第0个元素，第二行取第1个，第三行取第2个，去掉负号，即[0.7292,1.6778,1.8467],求平均值，就可以得到损失值。
# (0.7292+1.6778+1.8467)/3 = 1.4179


loss=torch.nn.NLLLoss()
target=torch.tensor([0,1,2])
nll_out = loss(log_output,target)
print('nll_out:\n',nll_out)

