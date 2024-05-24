# 获取用户输入
name = input("请输入你的名字: ")
age = input("请输入你的年龄: ")

# 尝试将年龄转换为整数
try:
    age = int(age)
except ValueError:
    print("输入的年龄不正确，请输入一个数字。")
else:
    # 根据年龄给出不同的问候
    if age < 18:
        greeting = f"你好，{name}! 你是一个青少年。"
    else:
        greeting = f"欢迎，{name}! 你是一个成年人。"
    
    print(greeting)
