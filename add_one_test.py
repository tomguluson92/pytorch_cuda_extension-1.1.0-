# -*- coding: utf-8 -*-
import torch

import add_one_cuda

a = torch.randn(1, 3, 2, 2).cuda()
print(a)
print(add_one_cuda.test1(a, 5))
