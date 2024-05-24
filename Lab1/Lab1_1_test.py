import sys
import platform
import numpy as np

def print_versions():
    print("Python version:", sys.version)
    print("System version:", platform.system(), platform.release())
    print("NumPy version:", np.__version__)

if __name__ == "__main__":
    print_versions()