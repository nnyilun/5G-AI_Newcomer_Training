# 从控制台读取输入
input_str = input("请输入两个以逗号分隔的整数X和Y: ")
# 分割字符串获取X和Y的值
X, Y = map(int, input_str.split(','))

# 初始化二维数组
array = []

# 外循环遍历行
for i in range(X):
    # 每行内部的列表
    row = []
    # 内循环遍历列
    for j in range(Y):
        # 计算元素值 i*j 并添加到行列表中
        row.append(i * j)
    # 将完整的行添加到数组中
    array.append(row)

# 打印结果
print(array)
