import torch


x = torch.arange(0, 5, 1)
y = torch.arange(0,3, 1)
x_grid, y_grid = torch.meshgrid(x, y, indexing="xy")
print(x_grid, y_grid)
x_grid = x_grid.unsqueeze(0).unsqueeze(0)
y_grid = y_grid.unsqueeze(0).unsqueeze(0)
xy_grid = torch.stack((x_grid, y_grid), dim=4)
xy_grid = xy_grid.repeat(3, 1, 1, 1, 1)*1000
print(xy_grid)