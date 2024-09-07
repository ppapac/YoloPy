import torch


x = torch.rand([2, 2])
y = torch.arange(0, 5, 1)

print(x)
print(y)
a = [x, x, x, y]

print(a[..., 0])
