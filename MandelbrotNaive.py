import matplotlib.pyplot as plt
import time, statistics

def linespace(start, end, num): #evenly spaced numbers between start and end
    if num <= 1:
        return [start]
    
    step = (end - start) / (num - 1)
    return [start + step * i for i in range(num)]

def mandelbrot_iteration(c, max_iter): 
    z = 0
    for n in range(max_iter):
        z = z * z + c          
        if abs(z) > 2:         
            return n
    return max_iter


def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    x_value = linespace(xmin, xmax, width)
    y_value = linespace(ymin, ymax, height)
    return (x_value, y_value, [[mandelbrot_iteration(complex(r, i), max_iter) for r in x_value] for i in y_value])


#set parameters for the Mandelbrot set
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