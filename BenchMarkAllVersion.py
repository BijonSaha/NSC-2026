import numpy as np 
import time, statistics
from numba import njit
from MandelbrotNaive import mandelbrot_set as mandelbrot_naive
from MandelbrotNumpy import mandelbrot_set as mandelbrot_numpy  


@njit
def mandelbrot_naive_numba(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    result = np.zeros((height, width), dtype=np.int32)

    for i in range(height):
        for j in range(width):
            c = x[j] + 1j * y[i]
            z = 0j
            n = 0
            while n < max_iter and z.real * z.real + z.imag * z.imag <= 4.0:
                z = z * z + c
                n += 1
            result[i, j] = n

    return result

def bench(fn, *args, runs=5):
    fn(*args)       # extra warm-up
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


if __name__ == "__main__":
    _ = mandelbrot_naive_numba(-2, 1, -1.5, 1.5, 64, 64)

    args = (-2, 1, -1.5, 1.5, 1024, 1024, 100)

    t_naive = bench(mandelbrot_naive,       *args)
    t_numpy = bench(mandelbrot_numpy,       *args)
    t_numba = bench(mandelbrot_naive_numba, *args)

    print(f"Naive: {t_naive:.3f}s")
    print(f"NumPy: {t_numpy:.3f}s  ({t_naive/t_numpy:.1f}x)")
    print(f"Numba: {t_numba:.3f}s  ({t_naive/t_numba:.1f}x)")