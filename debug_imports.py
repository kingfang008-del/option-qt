import sys
import os

try:
    import py_vollib_vectorized
    print(f"py_vollib_vectorized: {py_vollib_vectorized.__file__}")
except ImportError as e:
    print(f"FAILED to import py_vollib_vectorized: {e}")

try:
    import py_lets_be_rational
    print(f"py_lets_be_rational: {py_lets_be_rational.__file__}")
except ImportError as e:
    print(f"FAILED to import py_lets_be_rational: {e}")

try:
    import py_vollib
    print(f"py_vollib: {py_vollib.__file__}")
except ImportError as e:
    print(f"FAILED to import py_vollib: {e}")

print("\nFull sys.path:")
for p in sys.path:
    print(p)
