import numpy as np
from dataclasses import dataclass

G = 1.0
M = 1.0

@dataclass
class OrbitConfig:
    n_orbits: int = 16
    a_min: float = 0.5
    a_max: float = 5.0
    e_max: float = 0.3
    n_periods: float = 5.0         # number of periods to simulate
    dt_sim: float = 1e-3           # fine integrator dt
    dt_obs: float = 5e-2           # observation dt

def kepler_period(a):
    # For GM=1, T = 2π a^(3/2)
    return 2.0 * np.pi * (a ** 1.5)

def symplectic_step(r, v, dt):
    """Simple velocity-Verlet step for central force F = -GM r / |r|^3"""
    r_norm = np.linalg.norm(r, axis=-1, keepdims=True)
    acc = -G * M * r / (r_norm ** 3 + 1e-12)
    v_half = v + 0.5 * dt * acc
    r_new = r + dt * v_half
    r_norm_new = np.linalg.norm(r_new, axis=-1, keepdims=True)
    acc_new = -G * M * r_new / (r_norm_new ** 3 + 1e-12)
    v_new = v_half + 0.5 * dt * acc_new
    return r_new, v_new

def simulate_orbit(a, e, cfg: OrbitConfig):
    """Single 2D orbit with semi-major axis a and eccentricity e."""
    # periapsis distance:
    r_peri = a * (1 - e)
    r0 = np.array([r_peri, 0.0])
    # vis-viva: v^2 = GM (2/r - 1/a)
    v_mag = np.sqrt(G * M * (2.0 / r_peri - 1.0 / a))
    v0 = np.array([0.0, v_mag])

    T = kepler_period(a)
    t_final = cfg.n_periods * T

    n_steps = int(t_final / cfg.dt_sim)
    r = np.zeros((n_steps + 1, 2))
    v = np.zeros((n_steps + 1, 2))
    t = np.arange(n_steps + 1) * cfg.dt_sim

    r[0] = r0
    v[0] = v0
    for i in range(n_steps):
        r[i+1], v[i+1] = symplectic_step(r[i], v[i], cfg.dt_sim)

    # Subsample to observation times
    obs_indices = np.arange(0, n_steps + 1, int(cfg.dt_obs / cfg.dt_sim))
    return t[obs_indices], r[obs_indices], v[obs_indices]

def generate_dataset(cfg: OrbitConfig, sigma=1e-2, rng=None):
    """Generate an ensemble of noisy orbits."""
    if rng is None:
        rng = np.random.default_rng()

    a_vals = np.exp(rng.uniform(np.log(cfg.a_min), np.log(cfg.a_max), size=cfg.n_orbits))
    e_vals = rng.uniform(0.0, cfg.e_max, size=cfg.n_orbits)
    all_t, all_r_clean, all_v_clean, all_r_obs = [], [], [], []

    R0 = np.median(a_vals)

    for a, e in zip(a_vals, e_vals):
        t, r_clean, v_clean = simulate_orbit(a, e, cfg)
        noise = rng.normal(0.0, sigma * R0, size=r_clean.shape)
        r_obs = r_clean + noise

        all_t.append(t)
        all_r_clean.append(r_clean)
        all_v_clean.append(v_clean)
        all_r_obs.append(r_obs)

    return dict(
        t=all_t,
        r_clean=all_r_clean,
        v_clean=all_v_clean,
        r_obs=all_r_obs,
        a=a_vals,
        e=e_vals,
        sigma=sigma,
    )

def train_val_test_split(data, train_frac=0.7, val_frac=0.15, rng=None):
    n = len(data["a"])
    if rng is None:
        rng = np.random.default_rng()
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    def subset(indices):
        return {
            k: [data[k][i] for i in indices] if k in ["t", "r_clean", "v_clean", "r_obs"]
            else data[k][indices] if isinstance(data[k], np.ndarray)
            else data[k]
            for k in data
        }

    return subset(train_idx), subset(val_idx), subset(test_idx)
