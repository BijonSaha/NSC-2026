import matplotlib.pyplot as plt
import numpy as np
import time, statistics

def mandelbrot_iteration(C, max_iter):
    Z = np.zeros_like(C, dtype=complex)       # Z array, same shape as C
    M = np.zeros(C.shape, dtype=int)          # iteration count array

    for n in range(max_iter):                 # only loop 3 remains
        mask = np.abs(Z) <= 2                 # boolean mask: unescaped points
        Z[mask] = Z[mask]**2 + C[mask]        # update only unescaped points
        M[mask] += 1                          # increment their iteration count

    return M


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

    M = mandelbrot_iteration(C, max_iter)      # pass entire C array at once
    return (x, y, M)


# Set parameters for the Mandelbrot set
x_min, x_max = -2.0, 1.0
y_min, y_max = -1.5, 1.5
width, height = 1024, 1024
max_iterations = 100


def display(xmin, xmax, ymin, ymax, width, height, max_iter):
    x, y, z = mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter)
    z_rotated = np.rot90(z, k=1)                       # rotate 90 degrees clockwise
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
    print(f"Image dimensions: {grid.shape[0]} rows × {grid.shape[1]} columns")

    #Validation: Compare Naive vs NumPy 
    from MandelbrotNaive import mandelbrot_set as mandelbrot_naive

    print("\nNaive version")
    _, _, naive_grid = mandelbrot_naive(-2, 1, -1.5, 1.5, 1024, 1024, 100)

    print("Numpy version")
    _, _, numpy_grid = mandelbrot_set(-2, 1, -1.5, 1.5, 1024, 1024, 100)

    # Convert naive (list of lists) to numpy array for comparison
    naive_result = naive_grid
    numpy_result = numpy_grid

    # CORRECT comparison using np.allclose()
    if np.allclose(naive_result, numpy_result):
        print("Results match!")
    else:p
        print("Results differ!")

    # Check where they differ
    diff = np.abs(naive_result - numpy_result)
    print(f"Max difference: {diff.max()}")
    print(f"Different pixels: {(diff > 0).sum()}")
    