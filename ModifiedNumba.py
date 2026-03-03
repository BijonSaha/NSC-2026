from numba import njit
import numpy as np
import time
import matplotlib.pyplot as plt

# Inner point function (used by mandelbrot_numba_typed)
@njit
def mandelbrot_point_numba(c, max_iter):
    z = 0j
    for n in range(max_iter):
        if z.real * z.real + z.imag * z.imag > 4.0:
            return n
        z = z * z + c
    return max_iter


# Numba version with dtype parameter
@njit
def mandelbrot_numba_typed(xmin, xmax, ymin, ymax, width, height, max_iter=100, dtype=np.float64):
    x = np.linspace(xmin, xmax, width).astype(dtype)
    y = np.linspace(ymin, ymax, height).astype(dtype)
    result = np.zeros((height, width), dtype=np.int32)

    for i in range(height):
        for j in range(width):
            c = x[j] + 1j * y[i]
            result[i, j] = mandelbrot_point_numba(c, max_iter)

    return result


# Warm up (triggers JIT compilation -- float16 not supported by Numba)
_ = mandelbrot_numba_typed(-2, 1, -1.5, 1.5, 64, 64, dtype=np.float32)
_ = mandelbrot_numba_typed(-2, 1, -1.5, 1.5, 64, 64, dtype=np.float64)

# Test float32 and float64 only (Numba does not support float16)
for dtype in [np.float32, np.float64]:
    t0 = time.perf_counter()
    mandelbrot_numba_typed(-2, 1, -1.5, 1.5, 1024, 1024, dtype=dtype)
    print(f"{dtype.__name__}: {time.perf_counter()-t0:.3f}s")

# float16 not supported by Numba — use numpy fallback for visual comparison
def mandelbrot_numpy_f16(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    x = np.linspace(xmin, xmax, width).astype(np.float16)
    y = np.linspace(ymin, ymax, height).astype(np.float16)
    result = np.zeros((height, width), dtype=np.int32)
    for i in range(height):
        for j in range(width):
            c = x[j] + 1j * y[i]
            z = 0j
            for n in range(max_iter):
                if z.real * z.real + z.imag * z.imag > 4.0:
                    result[i, j] = n
                    break
                z = z * z + c
            else:
                result[i, j] = max_iter
    return result

r16 = mandelbrot_numpy_f16(-2, 1, -1.5, 1.5, 1024, 1024)
r32 = mandelbrot_numba_typed(-2, 1, -1.5, 1.5, 1024, 1024, dtype=np.float32)
r64 = mandelbrot_numba_typed(-2, 1, -1.5, 1.5, 1024, 1024, dtype=np.float64)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, result, title in zip(axes, [r16, r32, r64], ['float16', 'float32', 'float64 (ref)']):
    ax.imshow(result, cmap='hot')
    ax.set_title(title); ax.axis('off')
plt.savefig('precision_comparison.png', dpi=150)
plt.show()

print(f"Max diff float32 vs float64: {np.abs(r32 - r64).max()}")
print(f"Max diff float16 vs float64: {np.abs(r16 - r64).max()}")
