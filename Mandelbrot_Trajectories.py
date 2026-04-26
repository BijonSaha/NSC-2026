import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
N        = 512
MAX_ITER = 1000
TAU      = 0.01

# --- Seahorse Valley grid ---
x = np.linspace(-0.7530, -0.7490, N)
y = np.linspace( 0.0990,  0.1030, N)

C64 = (x[np.newaxis, :] + 1j * y[:, np.newaxis]).astype(np.complex128)
C32 = C64.astype(np.complex64)

# --- Iterate both dtypes simultaneously ---
z32    = np.zeros_like(C32)
z64    = np.zeros_like(C64)
diverge = np.full((N, N), MAX_ITER, dtype=np.int32)
active  = np.ones((N, N), dtype=bool)

for k in range(MAX_ITER):
    if not active.any():
        break
    z32[active] = z32[active] ** 2 + C32[active]
    z64[active] = z64[active] ** 2 + C64[active]
    diff = (np.abs(z32.real.astype(np.float64) - z64.real)
          + np.abs(z32.imag.astype(np.float64) - z64.imag))
    newly          = active & (diff > TAU)
    diverge[newly] = k
    active[newly]  = False

# --- Observations ---
n_diverged = np.sum(diverge < MAX_ITER)
fraction   = n_diverged / (N * N)
print(f"Pixels where float32/float64 diverge before max_iter: "
      f"{n_diverged} / {N*N} ({100 * fraction:.2f}%)")

# --- Plot ---
plt.figure(figsize=(7, 6))
plt.imshow(diverge, cmap='plasma', origin='lower',
           extent=[-0.7530, -0.7490, 0.0990, 0.1030])
plt.colorbar(label='First divergence iteration')
plt.title(f'Trajectory divergence  (τ = {TAU},  max_iter = {MAX_ITER})')
plt.xlabel('Re(c)')
plt.ylabel('Im(c)')
plt.tight_layout()
plt.savefig('mandelbrot_trajectory_divergence.png', dpi=150)
plt.show()
