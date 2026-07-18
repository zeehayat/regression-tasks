"""
Generates the static figures embedded in level_1_data_explorer.md and
level_1_5_adaptive_optimizers.md.
Run with: python3 generate_level1_figures.py
Outputs land in figures/
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 11,
})

# ---------------------------------------------------------------------------
# Day 4: toy dataset, OLS line, dashed vertical residuals
# ---------------------------------------------------------------------------
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 3.0, 5.0])
beta0, beta1 = 1 / 3, 1.5
y_hat = beta0 + beta1 * x

fig, ax = plt.subplots(figsize=(6, 4.5))
xs = np.linspace(0.5, 3.5, 100)
ax.plot(xs, beta0 + beta1 * xs, color="#1f77b4", linewidth=2,
        label=r"OLS fit: $\hat{y} = 0.333 + 1.5x$")
ax.scatter(x, y, color="#d62728", zorder=5, s=70, label="True data points")
ax.scatter(x, y_hat, color="#1f77b4", zorder=5, s=40, marker="x")
for xi, yi, yhi in zip(x, y, y_hat):
    ax.plot([xi, xi], [yi, yhi], color="#555555", linestyle="--", linewidth=1.5)
    mid = (yi + yhi) / 2
    ax.annotate(f"e={yi-yhi:+.3f}", (xi + 0.06, mid), fontsize=9, color="#555555")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Day 4: OLS Line and Residuals on the Toy Dataset")
ax.legend(loc="upper left")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "day4_ols_residuals.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Day 7: Gaussian density for the MLE generative story, one observation marked
# ---------------------------------------------------------------------------
sigma2 = 0.16666666666666688 / 3  # SSR/n from the chapter's worked example
sigma = np.sqrt(sigma2)
mu = beta0 + beta1 * 2.0  # x_i = 2 -> predicted mean for point 2
y_obs = 3.0  # actual y_2

xs = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
pdf = (1 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-((xs - mu) ** 2) / (2 * sigma2))
f_at_obs = (1 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-((y_obs - mu) ** 2) / (2 * sigma2))

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(xs, pdf, color="#1f77b4", linewidth=2)
ax.fill_between(xs, pdf, alpha=0.08, color="#1f77b4")
ax.axvline(mu, color="#555555", linestyle=":", linewidth=1.5)
ax.set_ylim(0, max(pdf) * 1.22)
ax.annotate(r"predicted mean $x_i^T\hat\beta = 3.333$", (mu, max(pdf) * 1.12),
            ha="center", fontsize=9, color="#555555")
ax.plot([y_obs, y_obs], [0, f_at_obs], color="#d62728", linewidth=2)
ax.scatter([y_obs], [f_at_obs], color="#d62728", zorder=5, s=50)
ax.annotate(f"observed $y_2=3.0$\ndensity height = {f_at_obs:.3f}",
            (y_obs, f_at_obs), textcoords="offset points", xytext=(10, 10),
            fontsize=9, color="#d62728")
ax.set_xlabel("y")
ax.set_ylabel("probability density")
ax.set_title("Day 7: The Gaussian MLE Story for One Observation")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "day7_mle_gaussian.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Level 1.5 (Adaptive Optimizers): narrow valley (unscaled) vs. symmetric bowl (scaled)
# ---------------------------------------------------------------------------
def quadratic_bowl(t0, t1, a, b):
    return 0.5 * (a * t0 ** 2 + b * t1 ** 2)

def gd_path(a, b, lr, steps, start):
    path = [np.array(start, dtype=float)]
    theta = np.array(start, dtype=float)
    for _ in range(steps):
        grad = np.array([a * theta[0], b * theta[1]])
        theta = theta - lr * grad
        path.append(theta.copy())
    return np.array(path)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

t0 = np.linspace(-10, 10, 300)
t1 = np.linspace(-10, 10, 300)
T0, T1 = np.meshgrid(t0, t1)

# Narrow valley: eigenvalues 1 and 40 (high condition number, unscaled features)
a1, b1 = 40.0, 1.0
Z1 = quadratic_bowl(T0, T1, a1, b1)
path1 = gd_path(a1, b1, lr=0.0485, steps=40, start=(9, 9))
axes[0].contour(T0, T1, Z1, levels=25, cmap="Blues")
axes[0].plot(path1[:, 0], path1[:, 1], color="#d62728", marker="o", markersize=3, linewidth=1.5)
axes[0].scatter([0], [0], color="black", marker="*", s=100, zorder=5)
axes[0].set_title(r"Unscaled features: $\kappa = 40$" "\n(zig-zagging path)")
axes[0].set_xlabel(r"$\theta_1$")
axes[0].set_ylabel(r"$\theta_2$")

# Symmetric bowl: eigenvalues ~equal (low condition number, scaled features)
a2, b2 = 2.0, 1.8
Z2 = quadratic_bowl(T0, T1, a2, b2)
path2 = gd_path(a2, b2, lr=0.45, steps=12, start=(9, 9))
axes[1].contour(T0, T1, Z2, levels=25, cmap="Greens")
axes[1].plot(path2[:, 0], path2[:, 1], color="#d62728", marker="o", markersize=3, linewidth=1.5)
axes[1].scatter([0], [0], color="black", marker="*", s=100, zorder=5)
axes[1].set_title(r"Scaled features: $\kappa \approx 1.1$" "\n(direct path)")
axes[1].set_xlabel(r"$\theta_1$")
axes[1].set_ylabel(r"$\theta_2$")

fig.suptitle("Day 10: Condition Number Shapes the Gradient Descent Path")
fig.tight_layout()
fig.savefig(OUT / "adaptive_optimizers_condition_number.png", dpi=150)
plt.close(fig)

print("Wrote:")
for f in sorted(OUT.glob("*.png")):
    print(" -", f, f.stat().st_size, "bytes")
