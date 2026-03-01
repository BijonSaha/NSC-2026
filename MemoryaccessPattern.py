import numpy as np
import time

N = 5000
A = np.random.rand(N, N)


def row_sum (A, N): #compute the row sums of A
    sum = 0
    for i in range(N):
        s=np.sum(A[i,:])
    return sum


def column_sum (A, N):  #compute the column sums of A
    sum = 0
    for i in range(N):
        s=np.sum(A[:,i])
    return sum

t0 = time.perf_counter() #timing both loops
row_sum(A, N)
t_row = time.perf_counter() - t0

t0 = time.perf_counter()
column_sum(A, N)
t_col = time.perf_counter() - t0

print(f"Row sums:    {t_row:.4f}s")
print(f"Column sums: {t_col:.4f}s")
print(f"Column is {t_col/t_row:.2f}x slower than row")

