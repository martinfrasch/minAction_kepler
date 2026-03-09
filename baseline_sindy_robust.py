"""
Noise-robust SINDy comparison for the Kepler benchmark.

Runs three SINDy variants on the same Kepler data used by MAL:
  1. Vanilla SINDy (STLSQ on raw finite differences)
  2. GP-SINDy (Gaussian Process smoothing → then finite differences)
  3. Ensemble-SINDy (bagged STLSQ for robustness to noise)

All methods use the same radial basis library as MAL: [1/r^2, 1/r, r, 1, 1/r^3]
applied to radial acceleration: a_r = -sum(c_k * phi_k(r))

Comparison metrics:
  - Which method identifies 1/r^2 as dominant?
  - Coefficient accuracy (recovered GM vs true GM=1.0)
  - Robustness across 10 seeds
  - Computation time

Usage:
    python baseline_sindy_robust.py
    python baseline_sindy_robust.py --seeds 0-9
"""

import time
import json
import argparse
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from data_kepler import OrbitConfig, generate_dataset, train_val_test_split

BASIS_LABELS = ["1/r^2", "1/r", "r", "1", "1/r^3"]


def compute_radial_data(data, stride=10):
    """Convert noisy orbit data to radial acceleration vs basis functions.

    Uses wide-stencil second differences (same as MAL) to estimate
    radial acceleration, then projects onto the radial basis.

    Returns:
        r_mag: (N,) radial distances
        a_radial: (N,) radial accelerations (positive = attractive)
        phi_matrix: (N, 5) basis function values
    """
    dt_obs = data["t"][0][1] - data["t"][0][0]
    s = stride

    all_r_mag = []
    all_a_radial = []
    all_phi = []

    for r_obs_np in data["r_obs"]:
        r_obs = np.array(r_obs_np)
        T = len(r_obs)
        if T <= 2 * s:
            continue

        # Wide-stencil acceleration
        a_est = (r_obs[2*s:] - 2 * r_obs[s:-s] + r_obs[:-2*s]) / (s * dt_obs) ** 2
        r_mid = r_obs[s:-s]
        r_mag = np.linalg.norm(r_mid, axis=-1)
        r_hat = r_mid / (r_mag[:, None] + 1e-12)

        # Radial projection: a_radial = -a · r_hat (positive for attractive)
        a_radial = -np.sum(a_est * r_hat, axis=-1)

        # Basis functions
        eps = 1e-8
        phi = np.column_stack([
            1.0 / (r_mag**2 + eps),   # 1/r^2
            1.0 / (r_mag + eps),       # 1/r
            r_mag,                      # r
            np.ones_like(r_mag),        # 1
            1.0 / (r_mag**3 + eps),    # 1/r^3
        ])

        all_r_mag.append(r_mag)
        all_a_radial.append(a_radial)
        all_phi.append(phi)

    return (np.concatenate(all_r_mag),
            np.concatenate(all_a_radial),
            np.concatenate(all_phi))


def compute_radial_data_gp(data, stride=10, length_scale=1.0, max_pts_per_orbit=200):
    """GP-smoothed version: smooth positions with GP before computing derivatives.

    Fits a GP to each orbit's x and y coordinates independently,
    then computes acceleration from the smoothed trajectories.
    Subsamples long orbits to max_pts_per_orbit for GP tractability.
    """
    dt_obs = data["t"][0][1] - data["t"][0][0]
    s = stride

    # GP kernel: smooth RBF + noise
    kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=1e-4)

    all_r_mag = []
    all_a_radial = []
    all_phi = []

    for r_obs_np, t_np in zip(data["r_obs"], data["t"]):
        r_obs = np.array(r_obs_np)
        t = np.array(t_np)
        T = len(r_obs)
        if T <= 2 * s:
            continue

        # Subsample for GP fitting (GP is O(n^3))
        if T > max_pts_per_orbit:
            sub_stride = T // max_pts_per_orbit
            fit_idx = np.arange(0, T, sub_stride)
        else:
            fit_idx = np.arange(T)

        # Smooth each coordinate with GP (fit on subsample, predict on all)
        r_smooth = np.zeros_like(r_obs)
        t_col_fit = t[fit_idx, None]
        t_col_all = t[:, None]
        for dim in range(2):
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, alpha=1e-6)
            gp.fit(t_col_fit, r_obs[fit_idx, dim])
            r_smooth[:, dim] = gp.predict(t_col_all)

        # Wide-stencil acceleration from smoothed positions
        a_est = (r_smooth[2*s:] - 2 * r_smooth[s:-s] + r_smooth[:-2*s]) / (s * dt_obs) ** 2
        r_mid = r_smooth[s:-s]
        r_mag = np.linalg.norm(r_mid, axis=-1)
        r_hat = r_mid / (r_mag[:, None] + 1e-12)

        a_radial = -np.sum(a_est * r_hat, axis=-1)

        eps = 1e-8
        phi = np.column_stack([
            1.0 / (r_mag**2 + eps),
            1.0 / (r_mag + eps),
            r_mag,
            np.ones_like(r_mag),
            1.0 / (r_mag**3 + eps),
        ])

        all_r_mag.append(r_mag)
        all_a_radial.append(a_radial)
        all_phi.append(phi)

    return (np.concatenate(all_r_mag),
            np.concatenate(all_a_radial),
            np.concatenate(all_phi))


def vanilla_sindy(phi, a_radial, threshold=0.05):
    """Vanilla SINDy: STLSQ on radial basis.

    Solves: a_radial ≈ phi @ coeffs, with sequential thresholding.
    """
    from sklearn.linear_model import LinearRegression

    n_basis = phi.shape[1]
    active = np.ones(n_basis, dtype=bool)
    coeffs = np.zeros(n_basis)

    for iteration in range(10):
        if not np.any(active):
            break
        reg = LinearRegression(fit_intercept=False)
        reg.fit(phi[:, active], a_radial)
        c = np.zeros(n_basis)
        c[active] = reg.coef_

        # Threshold
        small = np.abs(c) < threshold
        if np.all(active == ~small):
            break
        active = ~small
        coeffs = c

    # Final fit on active terms
    if np.any(active):
        reg = LinearRegression(fit_intercept=False)
        reg.fit(phi[:, active], a_radial)
        coeffs = np.zeros(n_basis)
        coeffs[active] = reg.coef_

    return coeffs


def ensemble_sindy(phi, a_radial, n_models=100, threshold=0.05, rng=None):
    """Ensemble-SINDy: bagged STLSQ for noise robustness.

    Runs STLSQ on n_models bootstrap samples, then takes median coefficients
    and inclusion probabilities.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_samples = len(a_radial)
    n_basis = phi.shape[1]
    all_coeffs = []

    for _ in range(n_models):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        c = vanilla_sindy(phi[idx], a_radial[idx], threshold=threshold)
        all_coeffs.append(c)

    all_coeffs = np.array(all_coeffs)

    # Inclusion probability: fraction of models where coefficient is nonzero
    inclusion_prob = np.mean(np.abs(all_coeffs) > 1e-10, axis=0)

    # Median coefficient (over nonzero entries only)
    median_coeffs = np.zeros(n_basis)
    for k in range(n_basis):
        nonzero = all_coeffs[:, k][np.abs(all_coeffs[:, k]) > 1e-10]
        if len(nonzero) > 0:
            median_coeffs[k] = np.median(nonzero)

    return median_coeffs, inclusion_prob


def run_comparison(seed=42):
    """Run all three SINDy variants + report results."""
    print(f"\n{'='*70}")
    print(f"SINDy Comparison (seed={seed})")
    print(f"{'='*70}")

    # Generate same data as MAL
    rng = np.random.default_rng(seed)
    cfg = OrbitConfig()
    data = generate_dataset(cfg, sigma=1e-2, rng=rng)
    rng_split = np.random.default_rng(seed)
    train_data, val_data, test_data = train_val_test_split(data, rng=rng_split)

    results = {}

    # --- Vanilla SINDy ---
    print("\n--- Vanilla SINDy (wide-stencil, s=10) ---")
    t0 = time.time()
    r_mag, a_radial, phi = compute_radial_data(train_data, stride=10)
    coeffs_vanilla = vanilla_sindy(phi, a_radial, threshold=0.05)
    time_vanilla = time.time() - t0

    dominant_v = int(np.argmax(np.abs(coeffs_vanilla)))
    print(f"  Coefficients: {dict(zip(BASIS_LABELS, ['%.4f' % c for c in coeffs_vanilla]))}")
    print(f"  Dominant: {BASIS_LABELS[dominant_v]} (c={coeffs_vanilla[dominant_v]:.4f})")
    print(f"  Time: {time_vanilla:.2f}s")
    results["vanilla"] = {
        "coefficients": coeffs_vanilla.tolist(),
        "dominant_basis": BASIS_LABELS[dominant_v],
        "dominant_idx": dominant_v,
        "GM_estimate": float(coeffs_vanilla[0]),
        "time_s": time_vanilla,
    }

    # --- GP-SINDy ---
    print("\n--- GP-SINDy (GP smoothing → wide-stencil, s=10) ---")
    t0 = time.time()
    r_mag_gp, a_radial_gp, phi_gp = compute_radial_data_gp(train_data, stride=10)
    coeffs_gp = vanilla_sindy(phi_gp, a_radial_gp, threshold=0.05)
    time_gp = time.time() - t0

    dominant_g = int(np.argmax(np.abs(coeffs_gp)))
    print(f"  Coefficients: {dict(zip(BASIS_LABELS, ['%.4f' % c for c in coeffs_gp]))}")
    print(f"  Dominant: {BASIS_LABELS[dominant_g]} (c={coeffs_gp[dominant_g]:.4f})")
    print(f"  Time: {time_gp:.2f}s")
    results["gp_sindy"] = {
        "coefficients": coeffs_gp.tolist(),
        "dominant_basis": BASIS_LABELS[dominant_g],
        "dominant_idx": dominant_g,
        "GM_estimate": float(coeffs_gp[0]),
        "time_s": time_gp,
    }

    # --- Ensemble-SINDy ---
    print("\n--- Ensemble-SINDy (100 bootstrap models, wide-stencil, s=10) ---")
    t0 = time.time()
    coeffs_ens, inclusion = ensemble_sindy(phi, a_radial, n_models=100, rng=np.random.default_rng(seed))
    time_ens = time.time() - t0

    dominant_e = int(np.argmax(np.abs(coeffs_ens)))
    print(f"  Median coefficients: {dict(zip(BASIS_LABELS, ['%.4f' % c for c in coeffs_ens]))}")
    print(f"  Inclusion prob:      {dict(zip(BASIS_LABELS, ['%.2f' % p for p in inclusion]))}")
    print(f"  Dominant: {BASIS_LABELS[dominant_e]} (c={coeffs_ens[dominant_e]:.4f})")
    print(f"  Time: {time_ens:.2f}s")
    results["ensemble"] = {
        "coefficients": coeffs_ens.tolist(),
        "inclusion_prob": inclusion.tolist(),
        "dominant_basis": BASIS_LABELS[dominant_e],
        "dominant_idx": dominant_e,
        "GM_estimate": float(coeffs_ens[0]),
        "time_s": time_ens,
    }

    # --- Also run with stride=1 (to show noise problem) ---
    print("\n--- Vanilla SINDy (naive, stride=1) [expected to fail] ---")
    t0 = time.time()
    r_mag_s1, a_radial_s1, phi_s1 = compute_radial_data(train_data, stride=1)
    coeffs_s1 = vanilla_sindy(phi_s1, a_radial_s1, threshold=0.05)
    time_s1 = time.time() - t0

    dominant_s1 = int(np.argmax(np.abs(coeffs_s1)))
    noise_std = np.std(a_radial_s1)
    signal_est = np.mean(np.abs(a_radial_s1))
    print(f"  Coefficients: {dict(zip(BASIS_LABELS, ['%.4f' % c for c in coeffs_s1]))}")
    print(f"  Dominant: {BASIS_LABELS[dominant_s1]} (c={coeffs_s1[dominant_s1]:.4f})")
    print(f"  a_radial noise_std={noise_std:.2f}, signal_est={signal_est:.2f}, SNR≈{signal_est/noise_std:.3f}")
    print(f"  Time: {time_s1:.2f}s")
    results["vanilla_naive"] = {
        "coefficients": coeffs_s1.tolist(),
        "dominant_basis": BASIS_LABELS[dominant_s1],
        "dominant_idx": dominant_s1,
        "noise_std": float(noise_std),
        "time_s": time_s1,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Noise-robust SINDy comparison")
    parser.add_argument("--seeds", type=str, default="42",
                        help="Seeds to run, e.g. '42' or '0-9'")
    args = parser.parse_args()

    if "-" in args.seeds and "," not in args.seeds:
        lo, hi = args.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    all_results = {}
    for seed in seeds:
        all_results[str(seed)] = run_comparison(seed)

    # Summary table across seeds
    if len(seeds) > 1:
        print(f"\n{'='*70}")
        print("SUMMARY ACROSS SEEDS")
        print(f"{'='*70}")

        for method in ["vanilla_naive", "vanilla", "gp_sindy", "ensemble"]:
            correct = 0
            gm_errors = []
            for seed in seeds:
                r = all_results[str(seed)][method]
                if r["dominant_idx"] == 0:
                    correct += 1
                if "GM_estimate" in r:
                    gm_errors.append(abs(r["GM_estimate"] - 1.0))

            method_name = {
                "vanilla_naive": "Vanilla SINDy (s=1)",
                "vanilla": "Vanilla SINDy (s=10)",
                "gp_sindy": "GP-SINDy (s=10)",
                "ensemble": "Ensemble-SINDy (s=10)",
            }[method]

            print(f"\n{method_name}:")
            print(f"  1/r^2 selected: {correct}/{len(seeds)}")
            if gm_errors:
                print(f"  |GM - 1.0| mean: {np.mean(gm_errors):.4f}")

    # Save
    out_path = "sindy_comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
