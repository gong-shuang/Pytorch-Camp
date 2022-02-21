# -*- coding:utf-8 -*-
"""
@file name  : hello pytorch.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2020-07-24
@brief      : 问世代码
"""

import torch

print("Hello World, Hello PyTorch {}".format(torch.__version__))

print("\nCUDA is available:{}, version is {}".format(torch.cuda.is_available(), torch.version.cuda))

print("\ndevice_name: {}".format(torch.cuda.get_device_name(0)))

print(torch.cuda.device)

print("out_t:")
out_t = torch.tensor((1,2))
print(out_t)

t=torch.zeros((3,3),out=out_t)
print(t)
print(out_t)

t1=torch.zeros_like(out_t)
print(t1)

w = torch.randn((1), requires_grad=True)
print(w)

w = torch.randn(20, 1)
# print(w)