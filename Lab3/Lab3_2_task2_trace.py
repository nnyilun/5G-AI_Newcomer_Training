import numpy as np

# 创建一个方阵
A = np.array([[1, 2], [3, 4]])

# 使用 einsum 计算迹
trace = np.einsum('ii', A)
print("Trace of the matrix:", trace)
