def fibonacci_recursive(n):
    if n <= 1:
        return n
    else:
        return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

def fibonacci_iterative(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

print("递归方法输出: ")
for i in range(10):
    print(fibonacci_recursive(i), end=' ')

print("\n迭代方法输出: ")
for i in range(10):
    print(fibonacci_iterative(i), end=' ')
print("")
