import sys
import platform

# Add your project packages here
import numpy as np
import pandas as pd
import matplotlib
import sklearn
import numba
import pytest, pytest_cov
import dask, distributed



print("Python version:", sys.version)
print("Platform:", platform.platform())

print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("Matplotlib version:", matplotlib.__version__)
print("Scikit-learn version:", sklearn.__version__)
print("Numba version:", numba.__version__)
print("Pytest version:", pytest.__version__)
print("Pytest-cov version:", pytest_cov.__version__)
print("Dask version:", dask.__version__)
print("Distributed version:", distributed.__version__)  