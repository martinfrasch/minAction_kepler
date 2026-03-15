#!/usr/bin/env python3
"""
Basis library sensitivity experiment for MAL manuscript.

Tests MAL robustness to confounding basis functions:
  Experiment 1: Add near-confounders (r^{-2.5}, r^{-1.5}) to the Kepler library
  Experiment 2: Remove the correct basis (r^{-2}) entirely
  Experiment 3: Expand to 8-term library with additional powers

For each experiment, runs 10 seeds and reports:
  - Correct-basis selection rate
  - Gate concentration C_gate (HHI)
  - Recovered force coefficient
  - Energy conservation sigma_H
  - Noetherian diagnostic pass/fail

Usage:
    python run_basis_sensitivity.py --experiment all --seeds 10
    python run_basis_sensitivity.py --experiment confounders --seeds 10
    python run_basis_sensitivity.py --experiment missing --seeds 10
    python run_basis_sensitivity.py --experiment expanded --seeds 10

Results saved to: basis_sensitivity_results.json
"""

import argparse
import json
import time
import os
import numpy as np
import torch

# Import project modules
from data_kepler import generate_kepler_data
from minaction_model import MinActionNet, NoetherForceBasis, minaction_loss, calibrate_theta


def make_basis_library(experiment: str):
    """Return (basis_functions, basis_labels, correct_index) for each experiment."""
    if experiment == "standard":
        # Standard 5-term library (control)
        return [
            lambda r: r**(-2),
            lambda r: r**(-1),
            lambda r: r,
            lambda r: torch.ones_like(r),
            lambda r: r**(-3),
        ], ["r^{-2}", "r^{-1}", "r", "1", "r^{-3}"], 0

    elif experiment == "confounders":
        # 7-term library with near-confounders r^{-2.5} and r^{-1.5}
        return [
            lambda r: r**(-2),
            lambda r: r**(-1),
            lambda r: r,
            lambda r: torch.ones_like(r),
            lambda r: r**(-3),
            lambda r: r**(-2.5),   # near-confounder
            lambda r: r**(-1.5),   # near-confounder
        ], ["r^{-2}", "r^{-1}", "r", "1", "r^{-3}", "r^{-2.5}", "r^{-1.5}"], 0

    elif experiment == "missing":
        # 4-term library WITHOUT the correct r^{-2} basis
        return [
            lambda r: r**(-1),
            lambda r: r,
            lambda r: torch.ones_like(r),
            lambda r: r**(-3),
        ], ["r^{-1}", "r", "1", "r^{-3}"], None  # correct basis absent

    elif experiment == "expanded":
        # 8-term library with additional powers
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


def compute_hhi(gates, thetas):
    """Compute Herfindahl-Hirschman gate concentration index."""
    effective = (gates * thetas.abs()).detach().cpu().numpy()
    effective = effective / (effective.sum() + 1e-12)
    K = len(effective)
    hhi = (effective**2).sum()
    return float((K * hhi - 1) / (K - 1))


def run_single_seed(seed, basis_fns, basis_labels, correct_idx,
                    n_epochs=200, device='cuda'):
    """Train one MAL model and return results dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    K = len(basis_fns)
    print(f"  Seed {seed}, K={K} basis functions...")

    # Generate data (same as standard Kepler benchmark)
    train_data, val_data, test_data = generate_kepler_data(
        n_orbits=16, sigma=0.01, seed=seed
    )

    # Build model with custom basis library size
    # NOTE: This requires MinActionNet/NoetherForceBasis to accept
    # a custom basis_fns list. If the current implementation hardcodes
    # K=5, you'll need to modify minaction_model.py first.
    # See the docstring at top of this file for required changes.
    model = MinActionNet(
        basis_fns=basis_fns,
        K=K,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop (standard BGNO protocol)
    t_start = time.time()
    for epoch in range(1, n_epochs + 1):
        # Compute schedule
        if epoch <= 50:
            alpha_E = 0.01
            tau = 1.0
        else:
            alpha_E = 0.01 + (1.0 - 0.01) * (epoch - 50) / 150
            tau = 1.0 * (0.05 / 1.0) ** ((epoch - 50) / 150)

        model.force_basis.tau = tau

        for batch in train_data:
            optimizer.zero_grad()
            loss = minaction_loss(
                model, batch,
                alpha_I=1.0, alpha_E=alpha_E,
                potential='gravity'
            )
            loss.backward()
            optimizer.step()

    train_time = time.time() - t_start

    # Extract results
    with torch.no_grad():
        gates = model.force_basis.get_gates().cpu()
        thetas = model.force_basis.theta.cpu()
        dominant_idx = gates.argmax().item()
        dominant_gate = gates[dominant_idx].item()
        dominant_label = basis_labels[dominant_idx]

        # Gate concentration
        c_gate = compute_hhi(gates, thetas)

        # Check correctness
        if correct_idx is not None:
            correct = (dominant_idx == correct_idx)
        else:
            correct = None  # can't evaluate if correct basis is absent

        # Calibrate coefficient
        theta_cal = calibrate_theta(model, test_data, dominant_idx)

    # Energy conservation diagnostic (rollout)
    # sigma_H computed from 5-orbital-period rollout
    sigma_H = compute_sigma_H(model, test_data, device)

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
        'train_time_s': round(train_time, 1),
    }
    print(f"    -> {dominant_label} (gate={dominant_gate:.3f}, "
          f"C_gate={c_gate:.3f}, correct={correct})")
    return result


def compute_sigma_H(model, test_data, device):
    """Compute Hamiltonian variance over long-horizon rollout."""
    # Placeholder — implement using model.rollout() and orbital_energy()
    # This should match the existing evaluation in evaluate_minaction.py
    try:
        from evaluate_minaction import compute_energy_conservation
        return compute_energy_conservation(model, test_data, device)
    except ImportError:
        print("    WARNING: evaluate_minaction not available, sigma_H=NaN")
        return float('nan')


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
                print(f"    FAILED seed {seed}: {e}")
                results.append({'seed': seed, 'error': str(e)})

        # Summary
        valid = [r for r in results if 'error' not in r]
        if valid:
            n_correct = sum(1 for r in valid if r['correct'] is True)
            n_total = len(valid)
            mean_cgate = np.mean([r['c_gate'] for r in valid])
            print(f"\nSUMMARY: {n_correct}/{n_total} correct, "
                  f"mean C_gate={mean_cgate:.3f}")

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
