#%%
import torch

#%%
print(torch.__version__)
print(" ")

#%%
a = torch.cuda.is_available()
print(a)
print(" ")

#%%
v = torch.version.cuda
print(v)

#%%
t = torch.tensor([1, 2, 3])
t

#%%
t = t.cuda()
t

#%%
