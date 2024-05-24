import numpy as np

# 创建两个矩阵
A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [1, 3]])

# 使用 einsum 进行矩阵乘法
C = np.einsum('ij,jk->ik', A, B)
print("Matrix Multiplication Result:\n", C)
