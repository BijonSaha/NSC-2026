"""
Mandelbrot_GPU.py — L10 GPU milestones M1, M2
Device: Apple M4 Pro (no native fp64 in OpenCL)
"""

import statistics
import time

import matplotlib.pyplot as plt
import numpy as np
import pyopencl as cl

from NumbaJit import mandelbrot_serial as numba_f64

# ── Domain (course convention) ───────────────────────────────────────────────
X_MIN, X_MAX = -2.5, 1.0
Y_MIN, Y_MAX = -1.25, 1.25
MAX_ITER     = 100
N_BENCH      = 1024

# ── OpenCL kernel source ──────────────────────────────────────────────────────
KERNEL_F32 = """
__kernel void mandelbrot_f32(
    __global int *result,
    const float x_min, const float x_max,
    const float y_min, const float y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= N || row >= N) return;

    float c_real = x_min + col * (x_max - x_min) / (float)N;
    float c_imag = y_min + row * (y_max - y_min) / (float)N;

    float z_real = 0.0f, z_imag = 0.0f;
    int count = 0;
    while (count < max_iter && z_real*z_real + z_imag*z_imag <= 4.0f) {
        float tmp = z_real*z_real - z_imag*z_imag + c_real;
        z_imag    = 2.0f * z_real * z_imag + c_imag;
        z_real    = tmp;
        count++;
    }
    result[row * N + col] = count;
}
"""


def timed(fn, runs=3):
    """Return median wall time in seconds over `runs` calls."""
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


if __name__ == "__main__":

    # ── OpenCL setup ─────────────────────────────────────────────────────────
    ctx   = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    dev   = ctx.devices[0]
    prog   = cl.Program(ctx, KERNEL_F32).build()
    kernel = cl.Kernel(prog, "mandelbrot_f32")    # retrieved once, reused to avoid warning

    def gpu_f32_timed(test_N):
        """Run the GPU f32 kernel at grid size test_N; return median wall time (s)."""
        buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, test_N * test_N * 4)
        kernel.set_args(buf,
                        np.float32(X_MIN), np.float32(X_MAX),
                        np.float32(Y_MIN), np.float32(Y_MAX),
                        np.int32(test_N),  np.int32(MAX_ITER))
        cl.enqueue_nd_range_kernel(queue, kernel, (test_N, test_N), None)
        queue.finish()                             # warm-up for this size
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            cl.enqueue_nd_range_kernel(queue, kernel, (test_N, test_N), None)
            queue.finish()
            times.append(time.perf_counter() - t0)
        return statistics.median(times)

    print(f"Device : {dev.name}")
    print(f"CUs    : {dev.max_compute_units}")
    print(f"fp64   : {'cl_khr_fp64' in dev.extensions}\n")

    N = N_BENCH

    # ═══════════════════════════════════════════════════════════════════════
    # M1 — GPU float32 kernel at N = 1024
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("M1 — GPU float32 kernel  (N = 1024)")
    print("=" * 60)

    img_host = np.zeros((N, N), dtype=np.int32)
    img_dev  = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, img_host.nbytes)

    kernel.set_args(img_dev,
                    np.float32(X_MIN), np.float32(X_MAX),
                    np.float32(Y_MIN), np.float32(Y_MAX),
                    np.int32(N),       np.int32(MAX_ITER))

    # Warm-up — forces OpenCL JIT compilation before timing
    cl.enqueue_nd_range_kernel(queue, kernel, (N, N), None)
    queue.finish()

    # Timed runs (3 × median)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        cl.enqueue_nd_range_kernel(queue, kernel, (N, N), None)
        queue.finish()                             # wait for GPU before stopping clock
        times.append(time.perf_counter() - t0)

    t_gpu_f32 = statistics.median(times)

    cl.enqueue_copy(queue, img_host, img_dev)
    queue.finish()

    print(f"GPU float32  N={N}: {t_gpu_f32*1e3:.1f} ms  ({t_gpu_f32:.4f} s)")

    plt.figure(figsize=(7, 5))
    plt.imshow(img_host, cmap='hot', origin='lower',
               extent=[X_MIN, X_MAX, Y_MIN, Y_MAX])
    plt.colorbar(label='Iteration count')
    plt.title(f'GPU Mandelbrot float32  (N={N},  {t_gpu_f32*1e3:.1f} ms)')
    plt.xlabel('Re(c)')
    plt.ylabel('Im(c)')
    plt.tight_layout()
    plt.savefig('mandelbrot_gpu_f32.png', dpi=150)
    plt.show()
    print("Saved mandelbrot_gpu_f32.png\n")

    # ═══════════════════════════════════════════════════════════════════════
    # M2 — Float32 vs Float64
    # Apple M4 Pro: no cl_khr_fp64 → compare GPU f32 vs Numba f64
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("M2 — float32 vs float64 comparison")
    print("  (no cl_khr_fp64 on Apple M4 Pro → GPU f32 vs Numba f64)")
    print("=" * 60)

    numba_f64(64, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER)   # Numba f64 warm-up

    for test_N in [N_BENCH, 2048]:
        t_f32 = gpu_f32_timed(test_N)
        t_f64 = timed(lambda n=test_N: numba_f64(n, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER))
        ratio = t_f64 / t_f32
        print(f"  N={test_N:5d}:  GPU f32 = {t_f32*1e3:7.1f} ms  |"
              f"  Numba f64 = {t_f64*1e3:7.1f} ms  |  ratio = {ratio:.2f}x")
    print()
