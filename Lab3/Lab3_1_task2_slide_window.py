import numpy as np

data = np.random.normal(loc=50, scale=10, size=100)

# 计算滑动窗口的平均值
def moving_average(data, window_size):
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

window_size = 5
ma_data = moving_average(data, window_size)

# 找到最大增长和最大下降
changes = np.diff(data)
max_increase = np.max(changes)
max_decrease = np.min(changes)

std_dev = np.std(data)
mean = np.mean(data)

print("Generated Data:", data)
print("Moving Average with window size {}: {}".format(window_size, ma_data))
print("Maximum Increase: {:.2f}".format(max_increase))
print("Maximum Decrease: {:.2f}".format(max_decrease))
print("Standard Deviation of the Data Set: {:.2f}".format(std_dev))
print("Mean of the Data Set: {:.2f}".format(mean))
