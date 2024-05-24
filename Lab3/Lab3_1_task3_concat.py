import numpy as np

array1 = np.array([[1, 2, 3], [4, 5, 6]])
array2 = np.array([[7, 8, 9], [10, 11, 12]])

vertical_concat = np.concatenate((array1, array2), axis=0)
print("Vertically Concatenated:\n", vertical_concat)

horizontal_concat = np.concatenate((array1, array2), axis=1)
print("Horizontally Concatenated:\n", horizontal_concat)
