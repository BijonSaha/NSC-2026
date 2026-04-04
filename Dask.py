import numpy as np
import time, statistics, os
from numba import njit
from dask import delayed
from dask.distributed import Client, LocalCluster
import dask
import matplotlib.pyplot as plt


# ── Numba kernel (same as L04/L05) ────────────────────────────────────────────

@njit(cache=True)
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = z_imag = 0.0
    for i in range(max_iter):
        z_sq = z_real*z_real + z_imag*z_imag
        if z_sq > 4.0:
            return i
        z_real, z_imag = z_real*z_real - z_imag*z_imag + c_real, \
                         2.0*z_real*z_imag + c_imag
    return max_iter


@njit(cache=True)
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


# ── MP2 M1: Dask Mandelbrot — Local ───────────────────────────────────────────

def mandelbrot_dask(N, x_min, x_max, y_min, y_max,
                    max_iter=100, n_chunks=32):
    """Wrap mandelbrot_chunk with dask.delayed — one task per row-band.
    dask.compute() replaces pool.map() from L04/L05.
    """
    chunk_size = max(1, N // n_chunks)
    tasks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        tasks.append(delayed(mandelbrot_chunk)(
            row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    parts = dask.compute(*tasks)
    return np.vstack(parts)


if __name__ == '__main__':
    N, max_iter = 1024, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
    N_WORKERS = os.cpu_count()

    # Create LocalCluster
    cluster = LocalCluster(n_workers=N_WORKERS, threads_per_worker=1)
    client  = Client(cluster)
    print(f"Dashboard: {client.dashboard_link}\n")

    # Warm up Numba JIT in ALL workers before timing
    client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10))
    print("Numba JIT warm-up complete in all workers ✓")

    # Serial reference (for correctness check)
    ref = mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)

    # --- MP2 M1: correctness check ---
    result = mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_chunks=32)
    assert np.array_equal(ref, result), "MP2 M1 FAILED: Dask result does not match serial!"
    print("MP2 M1 passed: Dask result matches serial ✓")

    # --- MP2 M1: timing (3 runs, median) ---
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_chunks=32)
        times.append(time.perf_counter() - t0)
    t_dask = statistics.median(times)
    print(f"\nDask local (n_chunks=32, n_workers={N_WORKERS}): {t_dask:.3f}s")

    # --- Serial baseline (for speedup calculation) ---
    mandelbrot_serial(64, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)  # JIT warm-up
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times)
    print(f"Serial baseline: {t_serial:.3f}s")

    # --- MP2 M2: chunk size sweep ---
    # T1: single-chunk baseline for LIF (all work on 1 task)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_chunks=1)
        times.append(time.perf_counter() - t0)
    t1_chunk = statistics.median(times)

    # Sweep range: smaller than L05 because Dask has higher α per task
    n_chunks_list = [N_WORKERS // 2, N_WORKERS,
                     2 * N_WORKERS, 4 * N_WORKERS,
                     8 * N_WORKERS, 16 * N_WORKERS]
    n_chunks_list = [max(1, n) for n in n_chunks_list]

    print(f"\n--- MP2 M2: chunk sweep (n_workers={N_WORKERS}) ---")
    print(f"{'n_chunks':>9}  {'time (s)':>9}  {'vs 1x':>7}  {'speedup':>8}  {'LIF':>6}")
    print("-" * 50)

    sweep_chunks, sweep_times, sweep_lifs = [], [], []
    t_1x = None

    for n_chunks in n_chunks_list:
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_chunks=n_chunks)
            times.append(time.perf_counter() - t0)
        t_par = statistics.median(times)
        if t_1x is None:
            t_1x = t_par
        lif     = N_WORKERS * t_par / t1_chunk - 1
        speedup = t_serial / t_par
        vs      = "baseline" if t_par == t_1x else f"{t_par/t_1x:.2f}x"
        sweep_chunks.append(n_chunks)
        sweep_times.append(t_par)
        sweep_lifs.append(lif)
        print(f"{n_chunks:>9d}  {t_par:>9.3f}  {vs:>7}  {speedup:>8.2f}x  {lif:>6.3f}")

    # Record optimal
    best_idx       = sweep_times.index(min(sweep_times))
    n_chunks_opt   = sweep_chunks[best_idx]
    t_min          = sweep_times[best_idx]
    lif_min        = sweep_lifs[best_idx]
    print(f"\nOptimal: n_chunks={n_chunks_opt}, t_min={t_min:.3f}s, LIF_min={lif_min:.3f}")

    # Plot wall time vs n_chunks (log scale)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep_chunks, sweep_times, 'b-o')
    ax.axvline(n_chunks_opt, color='r', linestyle='--', label=f'Optimal n_chunks={n_chunks_opt}')
    ax.set_xscale('log')
    ax.set_xlabel('n_chunks (log scale)')
    ax.set_ylabel('Wall time (s)')
    ax.set_title(f'Dask Local: Wall Time vs n_chunks (n_workers={N_WORKERS})')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('dask_chunk_sweep.png', dpi=150)
    plt.show()
    print("Plot saved to dask_chunk_sweep.png")

    input("\nDashboard still live — open http://127.0.0.1:8787/status in your browser. Press Enter to close...")
    client.close()
    cluster.close()
