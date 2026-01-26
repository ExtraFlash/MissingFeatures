import torch
import torch.nn as nn


# x = torch.randn(100, 5)
# w = torch.randn(5)
# w = nn.Parameter(w)
#
# a = torch.einsum('bd, d -> b', x, w)
# print('y4es')

a = torch.einsum('bnd,bd->bn', nn.Parameter(torch.randn(24,70,128,device='cuda')), nn.Parameter(torch.randn(24,128,device='cuda')))
c = 0