"""
10-seed sweep for Hooke's law benchmark.

Runs MAL training on harmonic oscillator data across seeds 0-9,
extracts crystallization timescales, and validates:
  - Gate selects 'r' basis (index 2)
  - Period is amplitude-independent (p_hat ≈ 0)
  - Noetherian criterion identifies correct basis

Usage:
    python run_hooke_sweep.py                 # full sweep
    python run_hooke_sweep.py --analyze-only  # just analyze existing checkpoints
"""

import os
import sys
import argparse
import time
import subprocess
import json
import numpy as np

BASIS_LABELS = ["1/r^2", "1/r", "r", "1", "1/r^3"]

THRESHOLDS = {
    "onset": 10,
    "crystallized": 100,
    "frozen": 1000,
    "locked": 10000,
}


def detect_tsparse(ckpt_path):
    """Load a Hooke MAL checkpoint and extract sparsification milestones."""
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    epoch_gates = np.array(ckpt["epoch_gates"])
    n_epochs = len(epoch_gates)

    ratios = []
    dominants = []
    for ep in range(n_epochs):
        g = epoch_gates[ep]
        idx = np.argsort(g)[::-1]
        dominants.append(int(idx[0]))
        if g[idx[1]] > 1e-12:
            ratios.append(g[idx[0]] / g[idx[1]])
        else:
            ratios.append(float("inf"))
    ratios = np.array(ratios)

    milestones = {}
    for name, thresh in THRESHOLDS.items():
        hits = np.where(ratios >= thresh)[0]
        milestones[name] = int(hits[0]) + 1 if len(hits) > 0 else None

    onset_ep = milestones.get("onset")
    frozen_ep = milestones.get("frozen")
    if onset_ep is not None and frozen_ep is not None and frozen_ep > onset_ep:
        i0 = onset_ep - 1
        i1 = frozen_ep - 1
        growth_rate = (ratios[i1] / ratios[i0]) ** (1.0 / (i1 - i0))
    else:
        growth_rate = None

    span = None
    if onset_ep is not None and frozen_ep is not None:
        span = frozen_ep - onset_ep

    final_gates = epoch_gates[-1]
    dom_idx = int(np.argmax(final_gates))
    theta_final = np.array(ckpt.get("epoch_theta", [[0]*5])[-1])

    state_dict = ckpt.get("model_state_dict", {})
    if "force_basis.theta" in state_dict:
        theta_cal = np.array(state_dict["force_basis.theta"].cpu().tolist())
    else:
        theta_cal = theta_final

    schedule = ckpt.get("schedule", {
        "n_epochs": n_epochs, "warmup_epochs": 50, "tau_final": 0.05,
    })

    return {
        "ckpt_path": ckpt_path,
        "dominant_basis": BASIS_LABELS[dom_idx],
        "dominant_gate_idx": dom_idx,
        "theta_calibrated": float(theta_cal[dom_idx]),
        "hooke_p_hat": float(ckpt.get("hooke_p_hat", float("nan"))),
        "T_theory": float(ckpt.get("T_theory", float("nan"))),
        "T_measured_mean": float(ckpt.get("T_measured_mean", float("nan"))),
        "onset_epoch": onset_ep,
        "tsparse_epoch": milestones.get("crystallized"),
        "frozen_epoch": frozen_ep,
        "locked_epoch": milestones.get("locked"),
        "onset_to_frozen_span": span,
        "growth_rate": growth_rate,
        "n_epochs": schedule.get("n_epochs", n_epochs),
        "tau_final": schedule.get("tau_final", 0.05),
        "total_time_s": float(ckpt.get("total_time", 0)),
        "final_selectivity": float(ratios[-1]),
    }


# --- Noetherian energy diagnostic ---
def noetherian_diagnostic(ckpt_path, seed):
    """For each basis function, compute energy conservation quality.

    For each basis hypothesis k, we:
    1. Compute the potential shape V_k(r) from the radial force law
    2. Find the optimal coefficient c_opt = argmin Var(KE + c*V_k)
       via c_opt = -Cov(KE, V_k) / Var(V_k)
    3. Return Var(KE + c_opt*V_k) — the residual energy variance

    The correct basis gives lowest residual variance because its potential
    shape matches the true conservation law regardless of theta magnitude.
    """
    import torch
    from data_hooke import HookeOrbitConfig, generate_dataset_hooke, train_val_test_split

    # Regenerate same data
    rng = np.random.default_rng(seed)
    cfg = HookeOrbitConfig()
    data = generate_dataset_hooke(cfg, sigma=1e-2, rng=rng)
    rng_split = np.random.default_rng(seed)
    train_data, _, _ = train_val_test_split(data, rng=rng_split)

    # Basis functions: [1/r^2, 1/r, r, 1, 1/r^3]
    # Corresponding potential shapes V(r) where F_r = -dV/dr:
    #   phi=1/r^2 -> V = -1/r
    #   phi=1/r   -> V = -ln(r)
    #   phi=r     -> V = r^2/2
    #   phi=1     -> V = r
    #   phi=1/r^3 -> V = -1/(2*r^2)

    s = 10
    dt_obs = train_data["t"][0][1] - train_data["t"][0][0]

    energy_variances = []

    for k_basis in range(5):
        orbit_rel_vars = []
        for r_obs_np in train_data["r_obs"]:
            r_obs = np.array(r_obs_np)
            T_len = len(r_obs)
            if T_len <= 2 * s:
                continue

            r_mid = r_obs[s:-s]
            r_mag = np.linalg.norm(r_mid, axis=-1)

            v_est = (r_obs[2*s:] - r_obs[:-2*s]) / (2 * s * dt_obs)
            v_mag2 = np.sum(v_est ** 2, axis=-1)
            KE = 0.5 * v_mag2

            if k_basis == 0:
                V = -1.0 / (r_mag + 1e-8)
            elif k_basis == 1:
                V = -np.log(r_mag + 1e-8)
            elif k_basis == 2:
                V = r_mag**2 / 2.0
            elif k_basis == 3:
                V = r_mag
            elif k_basis == 4:
                V = -1.0 / (2.0 * r_mag**2 + 1e-8)

            # Per-orbit optimal coefficient: c_opt = -Cov(KE, V) / Var(V)
            cov_KE_V = np.mean((KE - np.mean(KE)) * (V - np.mean(V)))
            var_V = np.var(V)
            if var_V > 1e-12:
                c_opt = -cov_KE_V / var_V
            else:
                c_opt = 0.0

            E = KE + c_opt * V
            # Per-orbit relative energy variance
            E_mean = np.mean(E)
            if abs(E_mean) > 1e-12:
                rel_var = np.var(E) / E_mean**2
            else:
                rel_var = np.var(E)
            orbit_rel_vars.append(rel_var)

        # Average per-orbit relative variance
        energy_variances.append(float(np.mean(orbit_rel_vars)))

    return energy_variances


WORKER_SCRIPT = """
import sys, os
sys.path.insert(0, os.getcwd())

import numpy as np
from data_hooke import HookeOrbitConfig, generate_dataset_hooke, train_val_test_split
from train_hooke import main as train_hooke

seed = {seed}
save_path = "{save_path}"
n_epochs = {n_epochs}
warmup_epochs = {warmup_epochs}
tau_final = {tau_final}
k = {k}

rng_data = np.random.default_rng(seed)
cfg = HookeOrbitConfig(k=k)
data = generate_dataset_hooke(cfg, sigma=1e-2, rng=rng_data)
rng_split = np.random.default_rng(seed)
train_data, val_data, test_data = train_val_test_split(data, rng=rng_split)

train_hooke(
    save_path=save_path,
    data_splits=(train_data, val_data, test_data),
    seed=seed,
    n_epochs=n_epochs,
    warmup_epochs=warmup_epochs,
    tau_final=tau_final,
    quiet=True,
    k=k,
)
print(f"DONE seed={{seed}} -> {{save_path}}")
"""


def run_one_seed(seed, save_path, n_epochs=200, warmup_epochs=50, tau_final=0.05, k=1.0):
    script = WORKER_SCRIPT.format(
        seed=seed, save_path=save_path,
        n_epochs=n_epochs, warmup_epochs=warmup_epochs,
        tau_final=tau_final, k=k,
    )
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True, text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    elapsed = time.time() - t0
    return seed, save_path, result.returncode, elapsed, result.stdout, result.stderr


def tabulate_results(results, noether_results=None):
    print()
    print("=" * 140)
    print("HOOKE'S LAW SWEEP RESULTS")
    print("=" * 140)
    print(
        f"{'Label':<12s} | {'Basis':>6s} | {'theta':>7s} | {'p_hat':>7s} | "
        f"{'T_meas':>7s} | {'T_theo':>7s} | "
        f"{'Onset':>6s} | {'t_sparse':>8s} | {'Frozen':>6s} | "
        f"{'Span':>5s} | {'Growth':>7s} | {'Time(s)':>8s}"
    )
    print("-" * 140)

    for r in results:
        label = r.get("label", r["ckpt_path"])
        print(
            f"{label:<12s} | "
            f"{r['dominant_basis']:>6s} | "
            f"{r['theta_calibrated']:>7.4f} | "
            f"{r['hooke_p_hat']:>7.4f} | "
            f"{r['T_measured_mean']:>7.4f} | "
            f"{r['T_theory']:>7.4f} | "
            f"{str(r['onset_epoch']) if r['onset_epoch'] else '-':>6s} | "
            f"{str(r['tsparse_epoch']) if r['tsparse_epoch'] else '-':>8s} | "
            f"{str(r['frozen_epoch']) if r['frozen_epoch'] else '-':>6s} | "
            f"{str(r['onset_to_frozen_span']) if r['onset_to_frozen_span'] else '-':>5s} | "
            f"{'%.4f' % r['growth_rate'] if r['growth_rate'] else '-':>7s} | "
            f"{r['total_time_s']:>8.1f}"
        )

    print("=" * 140)

    # Summary
    basis_counts = {}
    p_hats = []
    for r in results:
        b = r["dominant_basis"]
        basis_counts[b] = basis_counts.get(b, 0) + 1
        p_hats.append(r["hooke_p_hat"])

    print(f"\nBasis selection: {basis_counts}")
    print(f"Period exponent p_hat: mean={np.mean(p_hats):.4f}, std={np.std(p_hats):.4f} (expected ~0)")

    r_count = basis_counts.get("r", 0)
    print(f"Success rate (gate selects 'r'): {r_count}/{len(results)}")

    # Noetherian diagnostic
    if noether_results:
        print(f"\nNoetherian Energy Diagnostic (relative energy variance per basis hypothesis):")
        print(f"{'Seed':<8s} | {'1/r^2':>10s} | {'1/r':>10s} | {'r':>10s} | {'1':>10s} | {'1/r^3':>10s} | {'Winner':>8s} | {'Reject gravity?':>15s}")
        print("-" * 95)
        noether_correct = 0
        noether_rejects_gravity = 0
        for seed_i, ev in noether_results.items():
            winner_idx = int(np.argmin(ev))
            winner = BASIS_LABELS[winner_idx]
            if winner_idx == 2:
                noether_correct += 1
            # Check if correct-family potentials (r, 1) dominate gravity-family (1/r², 1/r, 1/r³)
            hooke_best = min(ev[2], ev[3])  # r or 1
            gravity_best = min(ev[0], ev[1], ev[4])  # 1/r², 1/r, 1/r³
            rejects = gravity_best > 3.0 * hooke_best
            if rejects:
                noether_rejects_gravity += 1
            print(
                f"seed={seed_i:<3d} | "
                + " | ".join(f"{v:>10.4e}" for v in ev)
                + f" | {winner:>8s}"
                + f" | {'YES' if rejects else 'no':>15s}"
            )
        print(f"\nNoetherian 'r' = lowest variance: {noether_correct}/{len(noether_results)}")
        print(f"Noetherian rejects gravity family (3x margin): {noether_rejects_gravity}/{len(noether_results)}")


def main():
    parser = argparse.ArgumentParser(description="Hooke's law 10-seed sweep")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--seeds", type=str, default="0-9")
    parser.add_argument("--k", type=float, default=1.0)
    args = parser.parse_args()

    if "-" in args.seeds and "," not in args.seeds:
        lo, hi = args.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    runs = []
    for seed in seeds:
        runs.append({
            "seed": seed,
            "save_path": f"hooke_sweep_seed{seed}_mal.pt",
            "n_epochs": 200,
            "warmup_epochs": 50,
            "tau_final": 0.05,
            "label": f"seed={seed}",
        })

    if not args.analyze_only:
        print(f"Running {len(runs)} Hooke training jobs...")
        for i, run in enumerate(runs):
            print(f"[{i+1}/{len(runs)}] Training {run['label']} -> {run['save_path']}")
            seed, path, rc, elapsed, stdout, stderr = run_one_seed(
                run["seed"], run["save_path"],
                run["n_epochs"], run["warmup_epochs"], run["tau_final"],
                k=args.k,
            )
            status = "OK" if rc == 0 else f"FAIL (rc={rc})"
            print(f"  {status} in {elapsed:.1f}s")
            if rc != 0:
                print(f"  STDERR: {stderr[-500:]}")

    # Analyze
    print("\nAnalyzing checkpoints...")
    results = []
    noether_results = {}
    for run in runs:
        path = run["save_path"]
        if not os.path.exists(path):
            print(f"  SKIP {path}")
            continue
        try:
            info = detect_tsparse(path)
            info["label"] = run["label"]
            results.append(info)

            ev = noetherian_diagnostic(path, run["seed"])
            noether_results[run["seed"]] = ev
        except Exception as e:
            print(f"  ERROR {path}: {e}")

    if not results:
        print("No checkpoints found.")
        return

    tabulate_results(results, noether_results)

    # Save JSON
    out = {
        "sweep_results": [],
        "noetherian": {},
    }
    for r in results:
        sr = dict(r)
        sr["growth_rate"] = float(sr["growth_rate"]) if sr["growth_rate"] is not None else None
        out["sweep_results"].append(sr)
    for seed_i, ev in noether_results.items():
        out["noetherian"][str(seed_i)] = ev

    out_path = "hooke_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
