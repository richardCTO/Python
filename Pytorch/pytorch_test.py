#%%
import torch

x = torch.Tensor([5, 3])
y = torch.Tensor([2, 1])
print (x*y)

#%%
x = torch.zeros([2, 5])
print(x)

#%%
y = torch.rand([2, 5])
y
#%%
y = y.view([1, 10])
y

#%%
import torch

torch.cuda.is_available()

#%%
