import cProfile, pstats
from MandelbrotNaive import mandelbrot_set as mandelbrot_naive
from MandelbrotNumpy import mandelbrot_set as mandelbrot_numpy

# Profile the naive version and save to file
cProfile.run('mandelbrot_naive(-2, 1, -1.5, 1.5, 512, 512, 100)', 'naive_profile.prof')

# Profile the numpy version and save to file
cProfile.run('mandelbrot_numpy(-2, 1, -1.5, 1.5, 512, 512, 100)', 'numpy_profile.prof')

# Print top 10 functions for each profile
for name in ('naive_profile.prof', 'numpy_profile.prof'):
    print(f"\n{'='*60}")
    print(f"Profile: {name}")
    print('='*60)
    stats = pstats.Stats(name)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
