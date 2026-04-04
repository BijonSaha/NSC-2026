# mandelbrot_parallel.py  (Tasks 1-3 are one continuous script)
import numpy as np
from numba import njit
from multiprocessing import Pool
import time, os, statistics
import matplotlib.pyplot as plt




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
    dx = (x_max - x_min) / (N - 1)
    dy = (y_max - y_min) / (N - 1)
    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            out[r, col] = mandelbrot_pixel(x_min + col*dx, c_imag, max_iter)
    return out


def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)




# ── M2: Parallel wrapper ──────────────────────────────────────────────────────

def _worker(args):
    return mandelbrot_chunk(*args)


def mandelbrot_parallel(N, x_min, x_max, y_min, y_max,
                        max_iter=100, n_workers=4, n_chunks=None):
    # L05 M1: n_chunks decoupled from n_workers for better load balancing
    if n_chunks is None:
        n_chunks = n_workers
    chunk_size = max(1, N // n_chunks)
    chunks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end

    with Pool(processes=n_workers) as pool:
        pool.map(_worker, chunks)          # un-timed warm-up: Numba JIT in workers
        parts = pool.map(_worker, chunks)

    return np.vstack(parts)



if __name__ == '__main__':
    N, max_iter = 1024, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25

    # Warm-up: trigger Numba JIT in main process
    mandelbrot_serial(64, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)

    # --- L04 M1: verify serial matches NumbaJit serial output ---
    from NumbaJit import mandelbrot_serial as numba_serial
    ref_l03 = numba_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
    ref_new = mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
    assert np.array_equal(ref_l03, ref_new), "L04 M1 FAILED: serial does not match L03 output!"
    print("L04 M1 passed: serial matches L03 output ✓")

    # --- L04 M2: verify parallel matches serial ---
    result = mandelbrot_parallel(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_workers=4)
    assert np.array_equal(ref_new, result), "L04 M2 FAILED: parallel does not match serial!"
    print("L04 M2 passed: parallel matches serial ✓")

    # Serial baseline
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times)

    # --- L04 M3: benchmark ---
    print("\n--- L04 M3: benchmark ---")
    worker_counts, speedups, efficiencies = [], [], []

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
        worker_counts.append(n_workers)
        speedups.append(speedup)
        efficiencies.append(speedup / n_workers * 100)
        print(f"{n_workers:2d} workers: {t_par:.3f}s, speedup={speedup:.2f}x,  eff={speedup/n_workers*100:.0f}%")

    # --- Speedup plot ---
    ideal = [p for p in worker_counts]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(worker_counts, ideal, 'k--', label='Ideal (linear)')
    ax.plot(worker_counts, speedups, 'b-o', label='Actual speedup')
    ax.set_xlabel('Number of workers')
    ax.set_ylabel('Speedup')
    ax.set_title('Speedup vs Number of Workers (Mandelbrot 1024×1024)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('speedup_plot.png', dpi=150)
    plt.show()
    print("Plot saved to speedup_plot.png")

    # --- L05 M1: n_chunks = 4 x n_workers ---
    # Correctness check
    ref = mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
    for nw in [1, 2, 4]:
        res = mandelbrot_parallel(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter,
                                  n_workers=nw, n_chunks=4*nw)
        assert np.array_equal(ref, res), f"L05 M1 FAILED at n_workers={nw}!"
    print("\nL05 M1 correctness check passed ✓")

    # Re-run worker sweep with n_chunks = 4 x n_workers
    print("\n--- L05 M1: benchmark (n_chunks = 4 x n_workers) ---")
    worker_counts_m1, speedups_m1 = [], []
    for n_workers in range(1, os.cpu_count() + 1):
        n_chunks = 4 * n_workers
        chunk_size = max(1, N // n_chunks)
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
        worker_counts_m1.append(n_workers)
        speedups_m1.append(speedup)
        print(f"{n_workers:2d} workers: {t_par:.3f}s, speedup={speedup:.2f}x,  eff={speedup/n_workers*100:.0f}%")

    # Comparison plot: L04 M3 vs L05 M1
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(worker_counts, worker_counts, 'k--', label='Ideal (linear)')
    ax.plot(worker_counts, speedups, 'b-o', label='L04 M3 (1x chunks)')
    ax.plot(worker_counts_m1, speedups_m1, 'r-s', label='L05 M1 (4x chunks)')
    ax.set_xlabel('Number of workers')
    ax.set_ylabel('Speedup')
    ax.set_title('Speedup: 1x vs 4x chunks (Mandelbrot 1024×1024)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('speedup_plot_m1.png', dpi=150)
    plt.show()
    print("Plot saved to speedup_plot_m1.png")

    # --- L05 M2: optimal chunk size sweep ---
    BEST_WORKERS = os.cpu_count()       # set to your L04 best worker count
    multipliers  = [1, 2, 4, 8, 16]

    # T1: single-worker baseline for LIF calculation
    chunk_1 = [(0, N, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)]
    with Pool(processes=1) as pool:
        pool.map(_worker, chunk_1)      # warm-up
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            np.vstack(pool.map(_worker, chunk_1))
            times.append(time.perf_counter() - t0)
    t1_worker = statistics.median(times)

    print(f"\n--- L05 M2: chunk sweep (n_workers = {BEST_WORKERS}) ---")
    print(f"{'n_chunks':>12}  {'time (s)':>9}  {'vs 1x':>7}  {'LIF':>6}")
    print("-" * 42)

    t_1x = None
    for mult in multipliers:
        n_chunks = mult * BEST_WORKERS
        chunk_size = max(1, N // n_chunks)
        chunks, row = [], 0
        while row < N:
            end = min(row + chunk_size, N)
            chunks.append((row, end, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter))
            row = end
        with Pool(processes=BEST_WORKERS) as pool:
            pool.map(_worker, chunks)   # warm-up
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                np.vstack(pool.map(_worker, chunks))
                times.append(time.perf_counter() - t0)
        t_par = statistics.median(times)
        if t_1x is None:
            t_1x = t_par
        lif = BEST_WORKERS * t_par / t1_worker - 1
        vs = t_par / t_1x
        label = "baseline" if mult == 1 else f"{vs:.2f}x"
        print(f"{n_chunks:>5d} ({mult:2d}x)    {t_par:>8.3f}  {label:>7}  {lif:>6.3f}")




