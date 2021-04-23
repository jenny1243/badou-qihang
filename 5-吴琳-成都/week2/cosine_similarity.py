import numpy as np
import torch

# print(np.random.rand(3,4,5))#三个四行五列的随机矩阵
# print(np.zeros((3,4,5)))
#基于numpy实现余弦相似度计算
a = np.random.rand(5)
b = np.random.rand(5)

def consine_similarity(a,b):
    if a.shape!=b.shape :
        raise Exception("两个向量的维度不一致")
    return a.dot(b) / (np.sqrt(np.sum(a**2))*np.sqrt(np.sum(b**2)))

out = consine_similarity(a,b)
print(out)

#batch normalization层 前向计算
x = np.random.rand(6).reshape(3,2)
x_torch = torch.Tensor(x)
bn = torch.nn.BatchNorm1d(2)
print(bn(x_torch))
print(x - np.mean(x, axis=0))

def Batch_Normolization(x):
    return (x - np.mean(x, axis=0)) / (np.sqrt(np.var(x, axis=0)))

test = Batch_Normolization(x)
print(test)

