# use_calculator.py
from mathoperations import add, subtract, multiply, divide

def main():
    print("Add: 5 + 3 =", add(5, 3))
    print("Subtract: 5 - 3 =", subtract(5, 3))
    print("Multiply: 5 * 3 =", multiply(5, 3))
    print("Divide: 5 / 3 =", divide(5, 3))
    print("Divide by zero: 5 / 0 =", divide(5, 0))

if __name__ == "__main__":
    main()
