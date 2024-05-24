import numpy as np

# 创建两个 2x2 矩阵
A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [1, 3]])

# 矩阵乘法
product = np.dot(A, B)
print("Matrix Product:\n", product)

# 计算第一个矩阵的逆
inverse_A = np.linalg.inv(A)
print("Inverse of Matrix A:\n", inverse_A)
