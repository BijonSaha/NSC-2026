import matplotlib.pyplot as plt
import numpy as np
import time, statistics

def mandelbrot_iteration(c, max_iter):
    z = 0
    for n in range(max_iter):
        z = z * z + c          # compute z_{n+1} first
        if abs(z) > 2:         # then check if it escapes
            return n
    return max_iter


def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    # Create 1D arrays with linspace
    x = np.linspace(xmin, xmax, width)    # 1024 x-values
    y = np.linspace(ymin, ymax, height)   # 1024 y-values

    # Create 2D grid with meshgrid
    X, Y = np.meshgrid(x, y)              # 2D grids

    # Combine into complex array C
    C = X + 1j*Y                          # Complex grid

    # Verify shape and dtype
    print(f"Shape: {C.shape}")            # (1024, 1024)
    print(f"Type: {C.dtype}")             # complex128

    grid = [[mandelbrot_iteration(C[i, j], max_iter) for j in range(width)] for i in range(height)]
    return (x, y, grid)


# Set parameters for the Mandelbrot set
x_min, x_max = -2.0, 1.0
y_min, y_max = -1.5, 1.5
width, height = 1024, 1024
max_iterations = 100


def display(xmin, xmax, ymin, ymax, width, height, max_iter):
    x, y, z = mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter)
    z_rotated = [list(row) for row in zip(*z)][::-1]  # rotate 90 degrees clockwise
    plt.imshow(z_rotated, extent=(ymin, ymax, xmax, xmin), cmap='inferno', origin='lower')
    plt.colorbar(label='Iteration count')
    plt.xlabel('Im')
    plt.ylabel('Re')
    plt.title('Mandelbrot Set')
    plt.show()


# Call the display function to show the Mandelbrot set
if __name__ == "__main__":
    print("Computing Mandelbrot Set...")
    display(x_min, x_max, y_min, y_max, width, height, max_iterations)
    print("Done!")

    def benchmark(func, *args, n_runs=3):
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            result = func(*args)
            times.append(time.perf_counter() - t0)
        median_t = statistics.median(times)
        print(f"Median: {median_t:.4f}s (min={min(times):.4f}, max={max(times):.4f})")
        return median_t, result

    t, M = benchmark(mandelbrot_set, -2, 1, -1.5, 1.5, 1024, 1024, 100, n_runs=3)
    _, _, grid = M  # unpack the tuple
    print(f"\nResult type: {type(M)}")
    print(f"Image dimensions: {len(grid)} rows × {len(grid[0])} columns")
