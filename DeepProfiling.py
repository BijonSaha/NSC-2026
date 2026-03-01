import numpy as np

@profile  # Add this decorator — provided by kernprof, no import needed
def mandelbrot_naive(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    result = np.zeros((height, width), dtype=int)

    for i in range(height):
        for j in range(width):
            c = x[j] + 1j * y[i]
            z = 0
            for n in range(max_iter):
                if abs(z) > 2:
                    result[i, j] = n
                    break
                z = z * z + c
            else:
                result[i, j] = max_iter

    return result


if __name__ == "__main__":
    mandelbrot_naive(-2, 1, -1.5, 1.5, 512, 512)
