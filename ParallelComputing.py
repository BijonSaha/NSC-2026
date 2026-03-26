# mandelbrot_parallel.py  (Tasks 1-3 are one continuous script)
import numpy as np
from numba import njit
from multiprocessing import Pool
import time, os, statistics




@njit
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = z_imag = 0.0
    for i in range(max_iter):
        z_sq = z_real*z_real + z_imag*z_imag
        if z_sq > 4.0: return i
        z_real, z_imag = z_real*z_real - z_imag*z_imag + c_real, 2.0*z_real*z_imag + c_imag
    return max_iter


@njit
def mandelbrot_chunk(row_start, row_end, N,
                     x_min, x_max, y_min, y_max, max_iter):
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            out[r, col] = mandelbrot_pixel(x_min + col*dx, c_imag, max_iter)
    return out


def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)




def _worker(args):
    """Module-level unpacker — required for multiprocessing pickling."""
    return mandelbrot_chunk(*args)


def mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter, n_workers):
    chunk_size = max(1, N // n_workers)
    chunks, row = [], 0
    while row < N:
        end = min(row + chunk_size, N)
        chunks.append((row, end, N, x_min, x_max, y_min, y_max, max_iter))
        row = end
    with Pool(processes=n_workers) as pool:
        parts = pool.map(_worker, chunks)
    return np.vstack(parts)



if __name__ == "__main__":
    N, max_iter = 1024, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25

    # M1 warm-up: trigger Numba JIT in main process (excluded from timing)
    mandelbrot_serial(64, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)

    # M1 correctness: verify parallel matches serial
    ref = mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
    par = mandelbrot_parallel(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, 2)
    assert np.array_equal(ref, par), "M1 FAILED: parallel does not match serial!"
    print("M1 passed: parallel matches serial ✓\n")

    # Serial baseline (Numba already warm after M1 warm-up)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times)

    print(f"{'workers':>7}  {'time':>8}  {'speedup':>8}  {'efficiency':>10}")
    print("-" * 42)

    for n_workers in range(1, os.cpu_count() + 1):
        chunk_size = max(1, N // n_workers)
        chunks, row = [], 0
        while row < N:
            end = min(row + chunk_size, N)
            chunks.append((row, end, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter))
            row = end

        with Pool(processes=n_workers) as pool:
            pool.map(_worker, chunks)          # warm-up: Numba JIT in all workers
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                np.vstack(pool.map(_worker, chunks))
                times.append(time.perf_counter() - t0)

        t_par = statistics.median(times)
        speedup = t_serial / t_par
        print(f"{n_workers:>7d}  {t_par:>8.3f}s  {speedup:>8.2f}x  {speedup/n_workers*100:>9.0f}%")
