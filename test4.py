import torch

a = torch.randint(10, (16000, 64))
b = torch.randint(10, (20, 64))
print(a.shape)
print(b.shape)

c = torch.matmul(a, b.t())
print(c.shape)

encoding_indices = torch.argmin(c, dim=1).unsqueeze(1)
print(encoding_indices.shape)

# 广播
# c = torch.randint(10, (3,2))

# d = torch.sum(c ** 2, dim=1, keepdim=True)
# print(d)
#
# e = torch.randint(10, (6,2))
# f = torch.sum(e ** 2, dim=1)
# print(f)


# c = torch.sum(a ** 2, dim=1, keepdim=True) + torch.sum(b ** 2, dim=1)
# print(c.shape)

s = torch.range(0, 5)
print(s)
print(s[0:-2])