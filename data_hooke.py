"""
Synthetic data generation for Hooke's law benchmark: F = -k*r (harmonic oscillator).

Generates 2D elliptical (Lissajous) orbits using symplectic Velocity-Verlet
with Gaussian noise, mirroring data_kepler.py for the spring-force case.

Key difference from Kepler: period T = 2*pi/omega is *independent* of amplitude.
"""

import numpy as np
from dataclasses import dataclass

K_SPRING = 1.0  # spring constant
M_MASS = 1.0    # particle mass

@dataclass
class HookeOrbitConfig:
    n_orbits: int = 16
    A_min: float = 0.5       # min amplitude
    A_max: float = 5.0       # max amplitude
    e_max: float = 0.3       # max eccentricity (velocity perturbation)
    n_periods: float = 5.0   # number of periods to simulate
    dt_sim: float = 1e-3     # fine integrator dt
    dt_obs: float = 5e-2     # observation dt
    k: float = K_SPRING
    m: float = M_MASS


def hooke_period(k=K_SPRING, m=M_MASS):
    """Period of harmonic oscillator: T = 2*pi*sqrt(m/k). Independent of amplitude."""
    return 2.0 * np.pi * np.sqrt(m / k)


def symplectic_step_hooke(r, v, dt, k=K_SPRING, m=M_MASS):
    """Velocity-Verlet step for Hooke's law: F = -k*r (vector)."""
    acc = -k * r / m
    v_half = v + 0.5 * dt * acc
    r_new = r + dt * v_half
    acc_new = -k * r_new / m
    v_new = v_half + 0.5 * dt * acc_new
    return r_new, v_new


def simulate_orbit_hooke(A, ecc, cfg: HookeOrbitConfig):
    """Single 2D harmonic orbit with amplitude A and eccentricity ecc.

    Args:
        A: amplitude (semi-major axis of ellipse)
        ecc: eccentricity, controls axis ratio. ecc=0 gives circle,
             ecc>0 gives ellipse with semi-minor axis b = A*(1-ecc).
    """
    omega = np.sqrt(cfg.k / cfg.m)

    # Initial conditions at (A, 0) with velocity in y-direction
    r0 = np.array([A, 0.0])
    # For a perfect circle: v0 = [0, omega*A]
    # For eccentricity: scale vy to make elliptical orbit
    vy = omega * A * (1.0 - ecc)
    v0 = np.array([0.0, vy])

    T = hooke_period(cfg.k, cfg.m)
    t_final = cfg.n_periods * T

    n_steps = int(t_final / cfg.dt_sim)
    r = np.zeros((n_steps + 1, 2))
    v = np.zeros((n_steps + 1, 2))
    t = np.arange(n_steps + 1) * cfg.dt_sim

    r[0] = r0
    v[0] = v0
    for i in range(n_steps):
        r[i+1], v[i+1] = symplectic_step_hooke(r[i], v[i], cfg.dt_sim, cfg.k, cfg.m)

    # Subsample to observation times
    obs_stride = int(cfg.dt_obs / cfg.dt_sim)
    obs_indices = np.arange(0, n_steps + 1, obs_stride)
    return t[obs_indices], r[obs_indices], v[obs_indices]


def generate_dataset_hooke(cfg: HookeOrbitConfig, sigma=1e-2, rng=None):
    """Generate an ensemble of noisy harmonic orbits."""
    if rng is None:
        rng = np.random.default_rng()

    # Log-uniform amplitudes
    A_vals = np.exp(rng.uniform(np.log(cfg.A_min), np.log(cfg.A_max), size=cfg.n_orbits))
    e_vals = rng.uniform(0.0, cfg.e_max, size=cfg.n_orbits)

    all_t, all_r_clean, all_v_clean, all_r_obs = [], [], [], []
    R0 = np.median(A_vals)

    for A, ecc in zip(A_vals, e_vals):
        t, r_clean, v_clean = simulate_orbit_hooke(A, ecc, cfg)
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
        A=A_vals,
        e=e_vals,
        sigma=sigma,
        k=cfg.k,
    )


def train_val_test_split(data, train_frac=0.7, val_frac=0.15, rng=None):
    n = len(data["A"])
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


if __name__ == "__main__":
    cfg = HookeOrbitConfig()
    rng = np.random.default_rng(42)
    data = generate_dataset_hooke(cfg, sigma=1e-2, rng=rng)

    print(f"Generated {cfg.n_orbits} Hooke orbits")
    print(f"Amplitudes: {data['A']}")
    print(f"Eccentricities: {data['e']}")
    print(f"Period (all orbits): T = {hooke_period():.4f}")
    for i in range(min(3, len(data['r_obs']))):
        r = data['r_obs'][i]
        print(f"  Orbit {i}: {len(r)} timesteps, r_max={np.max(np.linalg.norm(r, axis=-1)):.3f}")
