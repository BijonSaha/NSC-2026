import numpy as np
import pytest
from dask.distributed import Client, LocalCluster

from NumbaJit import (
    mandelbrot_pixel as _numba_pixel,
    mandelbrot_chunk,
    mandelbrot_serial,
    _worker,
    mandelbrot_parallel,
)
from Dask import mandelbrot_chunk as dask_chunk, mandelbrot_serial as dask_serial

# ── Numba JIT warm-up (module level — before any test runs) ─────────────────
_numba_pixel(0.0, 0.0, 10)
mandelbrot_chunk(0, 4, 4, -2.0, 1.0, -1.5, 1.5, 10)
dask_chunk(0, 4, 4, -2.0, 1.0, -1.5, 1.5, 10)


# ── Pure Python reference (check-first convention, matches Numba) ─────────────
def mandelbrot_pixel_py(c: complex, max_iter: int) -> int:
    z = 0j
    for n in range(max_iter):
        if z.real * z.real + z.imag * z.imag > 4.0:
            return n
        z = z * z + c
    return max_iter


def mandelbrot_pixel_numba(c: complex, max_iter: int) -> int:
    return _numba_pixel(c.real, c.imag, max_iter)


# ── Known cases — analytically provable, check-first convention ─────────────
# (same as slide example: origin, far exterior, outside left tip)
KNOWN_CASES = [
    (0 + 0j,    100, 100),   # origin: z = 0 forever, never escapes
    (5.0 + 0j,  100,   1),   # far outside: escapes at n=1 (|5|^2 = 25 > 4)
    (-2.5 + 0j, 100,   1),   # outside left tip: escapes at n=1 (|-2.5|^2 = 6.25 > 4)
]

IMPLEMENTATIONS = [mandelbrot_pixel_py, mandelbrot_pixel_numba]


# ── Dask LocalCluster fixture (module-scoped: one client shared across tests) ─
@pytest.fixture(scope="module")
def dask_client():
    cluster = LocalCluster(n_workers=2, threads_per_worker=1, silence_logs=True)
    client = Client(cluster)
    # Warm up Numba JIT in all workers before any test submits work
    client.run(lambda: dask_chunk(0, 4, 4, -2.0, 1.0, -1.5, 1.5, 10))
    yield client
    client.close()
    cluster.close()


# ════════════════════════════════════════════════════════════════════════════
# Numba @njit — compiled function is behaviourally identical to Python version
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("impl", IMPLEMENTATIONS, ids=["python", "numba"])
@pytest.mark.parametrize("c, max_iter, expected", KNOWN_CASES)
def test_pixel_all(impl, c: complex, max_iter: int, expected: int) -> None:
    """Python and Numba implementations must agree on all known pixel values."""
    assert impl(c, max_iter) == expected


def test_result_in_range() -> None:
    """Escape count must always lie in [0, max_iter] for any input."""
    max_iter = 50
    for c in [0 + 0j, -1 + 0j, 2 + 2j, 0.5 + 0.5j, -0.7 + 0.1j]:
        result = mandelbrot_pixel_py(c, max_iter)
        assert 0 <= result <= max_iter


# ════════════════════════════════════════════════════════════════════════════
# Multiprocessing Pool — test the worker, not the Pool machinery
# ════════════════════════════════════════════════════════════════════════════

def test_multiprocessing_worker_isolation() -> None:
    """_worker unpacks args and calls mandelbrot_chunk — result must match direct call."""
    N, MAX_ITER = 32, 50
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.0, 1.0, -1.5, 1.5
    args = (0, N, N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER)
    direct = mandelbrot_chunk(*args)
    via_worker = _worker(args)
    assert np.array_equal(direct, via_worker)


def test_multiprocessing_grid_matches_serial() -> None:
    """Assembled parallel grid must be pixel-identical to the serial result."""
    N, MAX_ITER = 32, 50
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.0, 1.0, -1.5, 1.5
    ref = mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER)
    par = mandelbrot_parallel(N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER, n_workers=2)
    assert np.array_equal(ref, par)


# ════════════════════════════════════════════════════════════════════════════
# Dask — test the underlying compute function, not the scheduler
# ════════════════════════════════════════════════════════════════════════════

def test_dask_chunk_future(dask_client: Client) -> None:
    """client.submit(dask_chunk, ...) result must match dask_serial on a small grid."""
    N, MAX_ITER = 32, 50
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.0, 1.0, -1.5, 1.5
    ref = dask_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER)
    future = dask_client.submit(dask_chunk, 0, N, N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER)
    result = dask_client.gather(future)
    assert np.array_equal(result, ref)
