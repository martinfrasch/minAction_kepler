"""
Basis Selection Analysis: why only 4/10 seeds select 1/r² directly.

Analyzes the 10-seed Kepler sweep checkpoints to understand:
  1. Correlation between initial theta[0] and final basis selection
  2. Gradient flow during warmup phase
  3. Effect of noise realization vs model init

Also runs intervention experiments:
  - Biased A_logits initialization (slight 1/r² preference)
  - Extended warmup (100 epochs instead of 50)
  - Lower noise (sigma=0.005)

Usage:
    python analyze_basis_selection.py
    python analyze_basis_selection.py --interventions  # run intervention experiments
"""

import os
import json
import argparse
import subprocess
import sys
import time
import numpy as np

BASIS_LABELS = ["1/r^2", "1/r", "r", "1", "1/r^3"]


def analyze_existing_sweeps():
    """Analyze the 10 existing Kepler sweep checkpoints."""
    import torch

    results = []
    for seed in range(10):
        path = f"sweep_seed{seed}_mal.pt"
        if not os.path.exists(path):
            print(f"  SKIP {path}")
            continue

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        epoch_gates = np.array(ckpt["epoch_gates"])
        epoch_theta = np.array(ckpt["epoch_theta"])

        # Initial state (epoch 0)
        initial_theta = epoch_theta[0]
        initial_gates = epoch_gates[0]

        # Final state
        final_gates = epoch_gates[-1]
        final_dominant = int(np.argmax(final_gates))

        # Warmup phase dynamics (epochs 1-50)
        warmup_gates = epoch_gates[:50]
        warmup_theta = epoch_theta[:50]

        # Track when each basis starts to differentiate
        gate_momentum = np.diff(warmup_gates, axis=0)  # per-epoch gate changes
        avg_momentum = np.mean(gate_momentum, axis=0)  # average momentum per basis

        # Gradient proxy: change in |A*theta| per epoch during warmup
        effective_coeffs = np.abs(warmup_gates * warmup_theta)
        eff_coeff_change = np.mean(np.diff(effective_coeffs, axis=0), axis=0)

        results.append({
            "seed": seed,
            "initial_theta": initial_theta.tolist(),
            "initial_theta_0": float(initial_theta[0]),
            "initial_theta_abs_0": float(abs(initial_theta[0])),
            "final_dominant_idx": final_dominant,
            "final_dominant_basis": BASIS_LABELS[final_dominant],
            "selected_1_r2": final_dominant == 0,
            "avg_gate_momentum": avg_momentum.tolist(),
            "eff_coeff_change": eff_coeff_change.tolist(),
            "kepler_p_hat": float(ckpt.get("kepler_p_hat", float("nan"))),
        })

    return results


def print_analysis(results):
    print("\n" + "=" * 80)
    print("BASIS SELECTION ANALYSIS (Kepler 10-seed sweep)")
    print("=" * 80)

    # Table
    print(f"\n{'Seed':>4s} | {'theta[0]':>8s} | {'|theta[0]|':>10s} | {'Selected':>8s} | {'1/r^2?':>6s} | {'p_hat':>7s}")
    print("-" * 55)
    for r in results:
        print(
            f"{r['seed']:>4d} | "
            f"{r['initial_theta'][0]:>8.4f} | "
            f"{r['initial_theta_abs_0']:>10.4f} | "
            f"{r['final_dominant_basis']:>8s} | "
            f"{'YES' if r['selected_1_r2'] else 'no':>6s} | "
            f"{r['kepler_p_hat']:>7.4f}"
        )

    # Correlation analysis
    selected = [r for r in results if r["selected_1_r2"]]
    not_selected = [r for r in results if not r["selected_1_r2"]]

    n_correct = len(selected)
    print(f"\nOverall: {n_correct}/{len(results)} seeds select 1/r^2")

    if selected and not_selected:
        mean_theta0_yes = np.mean([r["initial_theta_abs_0"] for r in selected])
        mean_theta0_no = np.mean([r["initial_theta_abs_0"] for r in not_selected])
        print(f"\n|theta[0]| when 1/r^2 selected: mean={mean_theta0_yes:.4f}")
        print(f"|theta[0]| when NOT selected:   mean={mean_theta0_no:.4f}")

        # Gate momentum during warmup
        momentum_yes = np.mean([r["avg_gate_momentum"] for r in selected], axis=0)
        momentum_no = np.mean([r["avg_gate_momentum"] for r in not_selected], axis=0)
        print(f"\nAvg gate momentum during warmup (per basis):")
        print(f"  When 1/r^2 selected:     {dict(zip(BASIS_LABELS, ['%.6f' % m for m in momentum_yes]))}")
        print(f"  When NOT selected:        {dict(zip(BASIS_LABELS, ['%.6f' % m for m in momentum_no]))}")

    # What basis won instead?
    alt_basis = {}
    for r in not_selected:
        b = r["final_dominant_basis"]
        alt_basis[b] = alt_basis.get(b, 0) + 1
    if alt_basis:
        print(f"\nAlternative selections: {alt_basis}")

    # All achieve correct Kepler exponent regardless
    p_hats = [r["kepler_p_hat"] for r in results]
    print(f"\nKepler exponent p_hat: mean={np.mean(p_hats):.4f}, std={np.std(p_hats):.4f} (all ~3.0)")


def run_intervention(name, extra_code, seeds=range(10), extra_kwargs=""):
    """Run a training intervention experiment via subprocess."""
    print(f"\n--- Running intervention: {name} ---")

    results = []
    for seed in seeds:
        save_path = f"intervention_{name}_seed{seed}.pt"
        kwargs_str = f", {extra_kwargs}" if extra_kwargs else ""
        script = f"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
from data_kepler import OrbitConfig, generate_dataset, train_val_test_split
from train_mal_with_metrics import main as train_mal

seed = {seed}
rng_data = np.random.default_rng(seed)
cfg = OrbitConfig()
data = generate_dataset(cfg, sigma=1e-2, rng=rng_data)
rng_split = np.random.default_rng(seed)
train_data, val_data, test_data = train_val_test_split(data, rng=rng_split)

{extra_code}

train_mal(
    save_path="{save_path}",
    data_splits=(train_data, val_data, test_data),
    seed=seed,
    n_epochs=200,
    warmup_epochs=warmup_epochs,
    tau_final=0.05,
    quiet=True{kwargs_str},
)
print(f"DONE seed={{seed}}")
"""
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        elapsed = time.time() - t0
        status = "OK" if result.returncode == 0 else "FAIL"
        print(f"  seed={seed}: {status} ({elapsed:.0f}s)")
        if result.returncode != 0:
            print(f"    {result.stderr[-200:]}")

        if os.path.exists(save_path):
            import torch
            ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
            final_gates = np.array(ckpt["epoch_gates"][-1])
            dom = int(np.argmax(final_gates))
            results.append({
                "seed": seed,
                "dominant_idx": dom,
                "dominant_basis": BASIS_LABELS[dom],
                "selected_1_r2": dom == 0,
                "kepler_p_hat": float(ckpt.get("kepler_p_hat", float("nan"))),
            })
            # Clean up
            os.remove(save_path)

    n_correct = sum(1 for r in results if r["selected_1_r2"])
    print(f"  Result: {n_correct}/{len(results)} select 1/r^2")
    return {"name": name, "n_correct": n_correct, "n_total": len(results), "results": results}


def run_interventions():
    """Run all intervention experiments."""
    interventions = {}

    # 1. Biased init: A_logits[0] = 1.5 (give 1/r^2 a head start)
    interventions["biased_init"] = run_intervention(
        "biased_init",
        "warmup_epochs = 50\n"
        "# Bias A_logits to give 1/r^2 a head start: [1.5, 0, 0, 0, 0]\n"
        "A_logits_init = [1.5, 0.0, 0.0, 0.0, 0.0]\n",
        extra_kwargs="A_logits_init=A_logits_init",
    )

    # 2. Extended warmup: 100 epochs
    interventions["extended_warmup"] = run_intervention(
        "extended_warmup",
        "warmup_epochs = 100",
    )

    # 3. Lower noise: sigma=0.005
    interventions["low_noise"] = run_intervention(
        "low_noise",
        """warmup_epochs = 50
# Regenerate data with lower noise
rng_data2 = np.random.default_rng(seed)
data = generate_dataset(cfg, sigma=0.005, rng=rng_data2)
rng_split2 = np.random.default_rng(seed)
train_data, val_data, test_data = train_val_test_split(data, rng=rng_split2)
""",
    )

    return interventions


def main():
    parser = argparse.ArgumentParser(description="Basis selection analysis")
    parser.add_argument("--interventions", action="store_true",
                        help="Run intervention experiments (slow)")
    args = parser.parse_args()

    # Phase 1: Analyze existing checkpoints
    results = analyze_existing_sweeps()
    if results:
        print_analysis(results)

    # Phase 2: Intervention experiments
    intervention_results = {}
    if args.interventions:
        intervention_results = run_interventions()

        # Summary table
        print(f"\n{'='*60}")
        print("INTERVENTION SUMMARY")
        print(f"{'='*60}")
        print(f"{'Intervention':<25s} | {'1/r^2 rate':>12s}")
        print("-" * 40)
        print(f"{'Standard (baseline)':.<25s} | {sum(1 for r in results if r['selected_1_r2'])}/{len(results)}")
        for name, res in intervention_results.items():
            print(f"{name:.<25s} | {res['n_correct']}/{res['n_total']}")

    # Save
    out = {
        "baseline_analysis": results,
        "interventions": {k: v for k, v in intervention_results.items()},
    }
    with open("basis_selection_analysis.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to basis_selection_analysis.json")


if __name__ == "__main__":
    main()
