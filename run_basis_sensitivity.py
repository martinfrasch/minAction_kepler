#!/usr/bin/env python3
"""
Basis library sensitivity experiment for MAL manuscript.

Tests MAL robustness to confounding basis functions:
  standard:    {r⁻², r⁻¹, r, 1, r⁻³} (K=5) — control
  confounders: + r⁻²·⁵, r⁻¹·⁵ (K=7) — near-degenerate bases
  missing:     {r⁻¹, r, 1, r⁻³} (K=4) — correct basis absent
  expanded:    + r², r⁻⁴, ln(r) (K=8) — scaling with library size

For each experiment, runs 10 seeds and reports:
  - Correct-basis selection rate
  - Gate concentration C_gate (HHI)
  - Recovered force coefficient
  - Energy conservation sigma_H
  - Kepler exponent p_hat

Usage:
    python run_basis_sensitivity.py --experiment all --seeds 10 --device cuda

Results saved to: basis_sensitivity_results.json
"""

import argparse
import json
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data_kepler import OrbitConfig, generate_dataset, train_val_test_split
from minaction_model import MinActionNet, minaction_loss, calibrate_theta, orbital_energy


# ---------------------------------------------------------------------------
# Data helpers (same as train_mal_with_metrics.py)
# ---------------------------------------------------------------------------
class OrbitDataset(Dataset):
    def __init__(self, data, max_T=None):
        self.r_obs = data["r_obs"]
        self.a = data["a"]
        self.t = data["t"]
        self.max_T = max_T

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        r = self.r_obs[idx]
        t = self.t[idx]
        if self.max_T is not None and len(t) > self.max_T:
            r = r[:self.max_T]
            t = t[:self.max_T]
        dt_obs = t[1] - t[0]
        return (torch.tensor(r, dtype=torch.float32),
                torch.tensor(dt_obs, dtype=torch.float32))


def collate_fn(batch):
    r_list, dt_list = zip(*batch)
    r = torch.stack(r_list, dim=0)
    dt = dt_list[0]
    return r, dt


# ---------------------------------------------------------------------------
# Basis libraries
# ---------------------------------------------------------------------------
def make_basis_library(experiment: str):
    """Return (basis_functions, basis_labels, correct_index) for each experiment."""
    if experiment == "standard":
        return [
            lambda r: r**(-2),
            lambda r: r**(-1),
            lambda r: r,
            lambda r: torch.ones_like(r),
            lambda r: r**(-3),
        ], ["r^{-2}", "r^{-1}", "r", "1", "r^{-3}"], 0

    elif experiment == "confounders":
        return [
            lambda r: r**(-2),
            lambda r: r**(-1),
            lambda r: r,
            lambda r: torch.ones_like(r),
            lambda r: r**(-3),
            lambda r: r**(-2.5),
            lambda r: r**(-1.5),
        ], ["r^{-2}", "r^{-1}", "r", "1", "r^{-3}", "r^{-2.5}", "r^{-1.5}"], 0

    elif experiment == "missing":
        return [
            lambda r: r**(-1),
            lambda r: r,
            lambda r: torch.ones_like(r),
            lambda r: r**(-3),
        ], ["r^{-1}", "r", "1", "r^{-3}"], None

    elif experiment == "expanded":
        return [
            lambda r: r**(-2),
            lambda r: r**(-1),
            lambda r: r,
            lambda r: torch.ones_like(r),
            lambda r: r**(-3),
            lambda r: r**(2),
            lambda r: r**(-4),
            lambda r: torch.log(r),
        ], ["r^{-2}", "r^{-1}", "r", "1", "r^{-3}", "r^2", "r^{-4}", "ln(r)"], 0

    else:
        raise ValueError(f"Unknown experiment: {experiment}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_hhi(gates, thetas):
    """Compute Herfindahl-Hirschman gate concentration index."""
    effective = (gates * thetas.abs()).detach().cpu().numpy()
    total = effective.sum() + 1e-12
    shares = effective / total
    K = len(shares)
    hhi_raw = (shares**2).sum()
    return float((K * hhi_raw - 1) / max(K - 1, 1))


def compute_sigma_H(model, test_data, dt_obs, device):
    """Energy conservation std from model rollout on test orbits."""
    model.eval()
    all_sigma = []
    n_sub = max(1, int(round(dt_obs / model.dt)))
    with torch.no_grad():
        for r_obs_np in test_data["r_obs"]:
            r_obs = torch.tensor(r_obs_np, dtype=torch.float32).to(device)
            T = r_obs.shape[0]
            if T < 20:
                continue
            r = r_obs[0].unsqueeze(0)  # (1, 2)
            s_v = min(5, T - 1)
            v = ((r_obs[s_v] - r_obs[0]) / (s_v * dt_obs)).unsqueeze(0)
            energies = []
            for t_idx in range(min(T - 1, 100)):
                for _ in range(n_sub):
                    r, v = model.integrate_step(r, v)
                E = orbital_energy(r, v, potential='gravity')
                energies.append(E.item())
                r = r_obs[t_idx + 1].unsqueeze(0)
            if len(energies) > 1:
                all_sigma.append(np.std(energies))
    return float(np.mean(all_sigma)) if all_sigma else float('nan')


def fit_kepler_exponent(a_vals, T_vals):
    y = np.log(T_vals ** 2)
    x = np.log(a_vals)
    A = np.vstack([x, np.ones_like(x)]).T
    p_hat, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return p_hat


def estimate_period(r_clean, t):
    x = r_clean[:, 0] - np.mean(r_clean[:, 0])
    N = len(x)
    corr = np.correlate(x, x, mode="full")
    corr = corr[N - 1:]
    corr = corr / np.arange(N, 0, -1)
    corr = corr / (corr[0] + 1e-12)
    i = 1
    while i < N - 1 and corr[i] <= corr[i - 1]:
        i += 1
    while i < N - 1 and corr[i] <= corr[i + 1]:
        i += 1
    return t[i] - t[0]


# ---------------------------------------------------------------------------
# Single seed training
# ---------------------------------------------------------------------------
def run_single_seed(seed, basis_fns, basis_labels, correct_idx,
                    n_epochs=200, warmup_epochs=50, tau_final=0.05,
                    device='cuda'):
    """Train one MAL model with custom basis library and return results dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    K = len(basis_fns)
    print(f"  Seed {seed}, K={K} basis functions...")

    # Generate data
    rng = np.random.default_rng(seed)
    cfg = OrbitConfig()
    data = generate_dataset(cfg, sigma=1e-2, rng=rng)
    train_data, val_data, test_data = train_val_test_split(data, rng=rng)

    max_T = 200
    train_ds = OrbitDataset(train_data, max_T=max_T)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)

    dt_obs = train_data["t"][0][1] - train_data["t"][0][0]
    model_dt = dt_obs / 5.0

    # Build model with custom basis
    model = MinActionNet(dt=model_dt, basis_fns=basis_fns).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop (standard two-phase schedule)
    t_start = time.time()
    for epoch in range(1, n_epochs + 1):
        if epoch <= warmup_epochs:
            alpha_E = 0.01
            tau = 1.0
        else:
            progress = (epoch - warmup_epochs) / (n_epochs - warmup_epochs)
            alpha_E = 0.01 + 0.99 * progress
            tau = 1.0 * (tau_final / 1.0) ** progress
        model.force_basis.tau = tau

        model.train()
        for r_obs, dt_batch in train_loader:
            r_obs = r_obs.to(device)
            optimizer.zero_grad()
            loss, loss_dict = minaction_loss(
                model, r_obs, dt_batch.item(),
                alpha_I=1.0, alpha_E=alpha_E, potential='gravity'
            )
            loss.backward()
            optimizer.step()

        if epoch % 50 == 0:
            with torch.no_grad():
                gates = torch.softmax(model.force_basis.A_logits / tau, dim=0)
                dom = gates.argmax().item()
            print(f"    epoch {epoch}: gate[{dom}]={gates[dom]:.3f} "
                  f"({basis_labels[dom]}), loss={loss_dict['total']:.4e}")

    train_time = time.time() - t_start

    # Post-training calibration
    calibrate_theta(model, train_data, dt_obs)

    # Extract results
    with torch.no_grad():
        gates = torch.softmax(model.force_basis.A_logits / model.force_basis.tau, dim=0).cpu()
        thetas = model.force_basis.theta.cpu()
        dominant_idx = gates.argmax().item()
        dominant_gate = gates[dominant_idx].item()
        dominant_label = basis_labels[dominant_idx]
        c_gate = compute_hhi(gates, thetas)

        if correct_idx is not None:
            correct = (dominant_idx == correct_idx)
        else:
            correct = None

        theta_cal = thetas[dominant_idx].item()

    # Energy conservation
    sigma_H = compute_sigma_H(model, test_data, dt_obs, device)

    # Kepler exponent from clean test data
    T_vals = []
    for r_clean, t in zip(test_data["r_clean"], test_data["t"]):
        T_vals.append(estimate_period(np.array(r_clean), np.array(t)))
    T_vals = np.array(T_vals)
    p_hat = fit_kepler_exponent(test_data["a"], T_vals)

    result = {
        'seed': seed,
        'K': K,
        'dominant_basis': dominant_label,
        'dominant_idx': dominant_idx,
        'dominant_gate': round(dominant_gate, 4),
        'correct': correct,
        'theta_calibrated': round(float(theta_cal), 4),
        'c_gate': round(c_gate, 4),
        'sigma_H': round(float(sigma_H), 6),
        'p_hat': round(float(p_hat), 4),
        'train_time_s': round(train_time, 1),
        'all_gates': [round(g, 4) for g in gates.tolist()],
    }
    print(f"    -> {dominant_label} (gate={dominant_gate:.3f}, "
          f"C_gate={c_gate:.3f}, theta={theta_cal:.4f}, correct={correct})")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Basis library sensitivity experiment")
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['standard', 'confounders', 'missing', 'expanded', 'all'],
                        help='Which experiment to run')
    parser.add_argument('--seeds', type=int, default=10,
                        help='Number of random seeds')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs per seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    args = parser.parse_args()

    if args.experiment == 'all':
        experiments = ['standard', 'confounders', 'missing', 'expanded']
    else:
        experiments = [args.experiment]

    all_results = {}

    for exp_name in experiments:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {exp_name}")
        print(f"{'='*60}")

        basis_fns, basis_labels, correct_idx = make_basis_library(exp_name)
        print(f"Library ({len(basis_fns)} terms): {basis_labels}")
        if correct_idx is not None:
            print(f"Correct basis: {basis_labels[correct_idx]}")
        else:
            print("Correct basis: ABSENT from library")

        results = []
        for seed in range(args.seeds):
            try:
                r = run_single_seed(
                    seed, basis_fns, basis_labels, correct_idx,
                    n_epochs=args.epochs, device=args.device
                )
                results.append(r)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"    FAILED seed {seed}: {e}")
                results.append({'seed': seed, 'error': str(e)})

        # Summary
        valid = [r for r in results if 'error' not in r]
        if valid:
            if correct_idx is not None:
                n_correct = sum(1 for r in valid if r['correct'] is True)
                n_total = len(valid)
                print(f"\n  Correct basis rate: {n_correct}/{n_total}")
            mean_cgate = np.mean([r['c_gate'] for r in valid])
            mean_sigma = np.mean([r['sigma_H'] for r in valid
                                  if not np.isnan(r['sigma_H'])])
            dominant_counts = {}
            for r in valid:
                b = r['dominant_basis']
                dominant_counts[b] = dominant_counts.get(b, 0) + 1
            print(f"  Mean C_gate: {mean_cgate:.3f}")
            print(f"  Mean sigma_H: {mean_sigma:.6f}")
            print(f"  Dominant basis counts: {dominant_counts}")

        all_results[exp_name] = {
            'basis_labels': basis_labels,
            'correct_idx': correct_idx,
            'seeds': results,
        }

    # Save results
    outfile = 'basis_sensitivity_results.json'
    with open(outfile, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {outfile}")


if __name__ == '__main__':
    main()
