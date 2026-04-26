import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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



# MP3 M2 — Mandelbrot Sensitivity Map (Condition Number)


eps32 = float(np.finfo(np.float32).eps)
delta = np.maximum(eps32 * np.abs(C64), 1e-10)

# --- Escape count function ---
def escape_count(C, max_iter):
    z   = np.zeros_like(C)
    cnt = np.full(C.shape, max_iter, dtype=np.int32)
    esc = np.zeros(C.shape, dtype=bool)
    for k in range(max_iter):
        z[~esc]    = z[~esc] ** 2 + C[~esc]
        newly      = ~esc & (np.abs(z) > 2.0)
        cnt[newly] = k
        esc[newly] = True
    return cnt

# --- Compute base and perturbed escape counts ---
n_base    = escape_count(C64,         MAX_ITER).astype(float)
n_perturb = escape_count(C64 + delta, MAX_ITER).astype(float)

# --- Condition number approximation ---
dn    = np.abs(n_base - n_perturb)
kappa = np.where(n_base > 0, dn / (eps32 * n_base), np.nan)

# --- Observations ---
print(f"κ max (99th percentile): {np.nanpercentile(kappa, 99):.2e}")
print(f"Interior pixels (n = MAX_ITER): κ = 0 by definition (set to nan)")
print(f"Pixels with κ >= 1: {np.sum(kappa >= 1)} / {N*N}")

# --- Plot ---
cmap_k = plt.cm.hot.copy()
cmap_k.set_bad('0.25')
vmax = np.nanpercentile(kappa, 99)

plt.figure(figsize=(7, 6))
plt.imshow(kappa, cmap=cmap_k, origin='lower',
           extent=[-0.7530, -0.7490, 0.0990, 0.1030],
           norm=LogNorm(vmin=1, vmax=vmax))
plt.colorbar(label=r'$\kappa(c)$  (log scale,  $\kappa \geq 1$)')
plt.title(r'Condition number approx  $\kappa(c) = |\Delta n|\,/\,(\varepsilon_{32}\,n(c))$')
plt.xlabel('Re(c)')
plt.ylabel('Im(c)')
plt.tight_layout()
plt.savefig('mandelbrot_sensitivity.png', dpi=150)
plt.show()
