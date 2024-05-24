def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    try:
        return x / y
    except ZeroDivisionError:
        return "Cannot divide by zero!"

if __name__ == "__main__":
    # 测试代码
    print("Add: ", add(5, 3))
    print("Subtract: ", subtract(5, 3))
    print("Multiply: ", multiply(5, 3))
    print("Divide: ", divide(5, 3))
    print("Divide by zero: ", divide(5, 0))
