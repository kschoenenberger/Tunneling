import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# electron, energies in eV, lengths in nm
HB2M = 0.0381   # hbar^2 / 2m_e  [eV nm^2]
E0 = 10.0       # barrier height V_0
a = 0.3         # barrier width


def transmission(E):
    E = np.asarray(E, dtype=float)
    E = np.where(np.abs(E - E0) < 1e-6, E + 1e-6, E)
    q = np.sqrt((E0 - E + 0j) / HB2M)          # alpha for E<E0, i*k' for E>E0
    s2 = np.abs(np.sinh(q * a))**2              # sinh^2 -> sin^2 above barrier
    return 1 / (1 + E0**2 / (4 * E * np.abs(E0 - E)) * s2)


def wavefunction(E, x):
    if abs(E - E0) < 1e-6:
        E += 1e-6
    k = np.sqrt(E / HB2M)
    q = np.sqrt((E0 - E + 0j) / HB2M)
    # unknowns B, C, D, F with A = 1
    M = np.array([
        [1, -1, -1, 0],
        [-1j * k, q, -q, 0],
        [0, np.exp(-q * a), np.exp(q * a), -np.exp(1j * k * a)],
        [0, -q * np.exp(-q * a), q * np.exp(q * a), -1j * k * np.exp(1j * k * a)],
    ])
    rhs = np.array([-1, -1j * k, 0, 0])
    B, C, D, F = np.linalg.solve(M, rhs)
    return np.where(x < 0, np.exp(1j * k * x) + B * np.exp(-1j * k * x),
           np.where(x <= a, C * np.exp(-q * x) + D * np.exp(q * x),
                    F * np.exp(1j * k * x)))


E_grid = np.linspace(0.01, 3 * E0, 1000)
T_grid = transmission(E_grid)
x = np.linspace(-1.5, a + 1.5, 1500)
E_init = 5.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plt.subplots_adjust(bottom=0.2)

ax1.plot(E_grid / E0, T_grid, label="T")
ax1.plot(E_grid / E0, 1 - T_grid, label="R")
ax1.axvline(1, color="gray", ls=":")
pT, = ax1.plot([], [], "o", color="C0")
pR, = ax1.plot([], [], "o", color="C1")
ax1.set_xlabel("$E/V_0$")
ax1.set_ylim(-0.02, 1.02)
ax1.legend()

ax2.axvspan(0, a, color="gray", alpha=0.3, label="barrier")
line_re, = ax2.plot([], [], lw=0.8, label=r"Re $\psi$")
line_abs, = ax2.plot([], [], lw=2, label=r"$|\psi|^2$")
ax2.set_xlim(x[0], x[-1])
ax2.set_ylim(-2.2, 4.2)
ax2.set_xlabel("x [nm]")
ax2.legend(loc="upper right")

ax_s = plt.axes([0.15, 0.06, 0.7, 0.04])
slider = Slider(ax_s, "E [eV]", 0.05, 3 * E0, valinit=E_init)


def update(E):
    T = transmission(E)
    pT.set_data([E / E0], [T])
    pR.set_data([E / E0], [1 - T])
    psi = wavefunction(E, x)
    line_re.set_data(x, psi.real)
    line_abs.set_data(x, np.abs(psi)**2)
    ax1.set_title(f"$V_0$ = {E0} eV, a = {a} nm   |   T = {T:.4f},  R = {1 - T:.4f}")
    ax2.set_title(f"E = {E:.2f} eV")
    fig.canvas.draw_idle()


slider.on_changed(update)
update(E_init)
plt.show()
