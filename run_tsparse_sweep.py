"""
Sweep seeds 0-9 + one slow-anneal run, extract t_sparse, tabulate.

Runs MAL training in parallel using subprocess workers (one GPU process
at a time to avoid CUDA contention, but data generation is overlapped).

Usage:
    python run_tsparse_sweep.py                  # seeds 0-9 + slow anneal
    python run_tsparse_sweep.py --workers 2      # 2 parallel GPU workers
    python run_tsparse_sweep.py --analyze-only   # just tabulate existing checkpoints
"""

import os
import sys
import argparse
import time
import subprocess
import json
import numpy as np

# ---------------------------------------------------------------------------
# t_sparse auto-detection
# ---------------------------------------------------------------------------
BASIS_LABELS = ["1/r^2", "1/r", "r", "1", "1/r^3"]

THRESHOLDS = {
    "onset": 10,
    "crystallized": 100,     # t_sparse
    "frozen": 1000,
    "locked": 10000,
}


def detect_tsparse(ckpt_path):
    """
    Load a MAL checkpoint and extract sparsification milestones from epoch_gates.

    Returns dict with:
        seed, dominant_basis, dominant_gate_idx, theta_calibrated, kepler_p_hat,
        onset_epoch, tsparse_epoch, frozen_epoch, locked_epoch,
        onset_to_frozen_span, growth_rate, schedule, total_time
    """
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    epoch_gates = np.array(ckpt["epoch_gates"])
    n_epochs = len(epoch_gates)

    # Compute selectivity ratios per epoch
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

    # Find milestone epochs (1-indexed)
    milestones = {}
    for name, thresh in THRESHOLDS.items():
        hits = np.where(ratios >= thresh)[0]
        milestones[name] = int(hits[0]) + 1 if len(hits) > 0 else None

    # Growth rate: geometric mean of ratio[t+1]/ratio[t] in onset→frozen window
    onset_ep = milestones.get("onset")
    frozen_ep = milestones.get("frozen")
    if onset_ep is not None and frozen_ep is not None and frozen_ep > onset_ep:
        i0 = onset_ep - 1  # 0-indexed
        i1 = frozen_ep - 1
        growth_rate = (ratios[i1] / ratios[i0]) ** (1.0 / (i1 - i0))
    else:
        growth_rate = None

    # Span
    span = None
    if onset_ep is not None and frozen_ep is not None:
        span = frozen_ep - onset_ep

    # Final state
    final_gates = epoch_gates[-1]
    dom_idx = int(np.argmax(final_gates))
    theta_final = np.array(ckpt.get("epoch_theta", [[0]*5])[-1])

    # Calibrated theta from checkpoint
    state_dict = ckpt.get("model_state_dict", {})
    if "force_basis.theta" in state_dict:
        theta_cal = np.array(state_dict["force_basis.theta"].cpu().tolist())
    else:
        theta_cal = theta_final

    schedule = ckpt.get("schedule", {
        "n_epochs": n_epochs,
        "warmup_epochs": 50,
        "tau_final": 0.05,
    })

    return {
        "ckpt_path": ckpt_path,
        "dominant_basis": BASIS_LABELS[dom_idx],
        "dominant_gate_idx": dom_idx,
        "theta_calibrated": float(theta_cal[dom_idx]),
        "kepler_p_hat": float(ckpt.get("kepler_p_hat", float("nan"))),
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


# ---------------------------------------------------------------------------
# Worker: train one seed via subprocess (avoids CUDA context sharing issues)
# ---------------------------------------------------------------------------
WORKER_SCRIPT = """
import sys, os
sys.path.insert(0, os.getcwd())

import numpy as np
from data_kepler import OrbitConfig, generate_dataset, train_val_test_split
from train_mal_with_metrics import main as train_mal

seed = {seed}
save_path = "{save_path}"
n_epochs = {n_epochs}
warmup_epochs = {warmup_epochs}
tau_final = {tau_final}

# Generate data
rng_data = np.random.default_rng(seed)
cfg = OrbitConfig()
data = generate_dataset(cfg, sigma=1e-2, rng=rng_data)
rng_split = np.random.default_rng(seed)
train_data, val_data, test_data = train_val_test_split(data, rng=rng_split)

train_mal(
    save_path=save_path,
    data_splits=(train_data, val_data, test_data),
    seed=seed,
    n_epochs=n_epochs,
    warmup_epochs=warmup_epochs,
    tau_final=tau_final,
    quiet=True,
)
print(f"DONE seed={seed} -> {{save_path}}")
"""


def run_one_seed(seed, save_path, n_epochs=200, warmup_epochs=50, tau_final=0.05):
    """Launch a subprocess to train one seed. Returns (seed, save_path, returncode, elapsed)."""
    script = WORKER_SCRIPT.format(
        seed=seed,
        save_path=save_path,
        n_epochs=n_epochs,
        warmup_epochs=warmup_epochs,
        tau_final=tau_final,
    )
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    elapsed = time.time() - t0
    return seed, save_path, result.returncode, elapsed, result.stdout, result.stderr


def tabulate_results(results, title="t_sparse SWEEP RESULTS"):
    """Pretty-print the results table."""
    print()
    print("=" * 130)
    print(title)
    print("=" * 130)
    print(
        f"{'Label':<20s} | {'Basis':>6s} | {'theta':>7s} | {'p_hat':>7s} | "
        f"{'Onset':>6s} | {'t_sparse':>8s} | {'Frozen':>6s} | {'Locked':>6s} | "
        f"{'Span':>5s} | {'Growth':>7s} | {'tau_f':>6s} | {'Time(s)':>8s}"
    )
    print("-" * 130)

    for r in results:
        label = r.get("label", r["ckpt_path"])
        onset = r["onset_epoch"]
        tsparse = r["tsparse_epoch"]
        frozen = r["frozen_epoch"]
        locked = r["locked_epoch"]
        span = r["onset_to_frozen_span"]
        gr = r["growth_rate"]
        print(
            f"{label:<20s} | "
            f"{r['dominant_basis']:>6s} | "
            f"{r['theta_calibrated']:>7.4f} | "
            f"{r['kepler_p_hat']:>7.4f} | "
            f"{str(onset) if onset else '-':>6s} | "
            f"{str(tsparse) if tsparse else '-':>8s} | "
            f"{str(frozen) if frozen else '-':>6s} | "
            f"{str(locked) if locked else '-':>6s} | "
            f"{str(span) if span else '-':>5s} | "
            f"{f'{gr:.4f}' if gr else '-':>7s} | "
            f"{r['tau_final']:>6.3f} | "
            f"{r['total_time_s']:>8.1f}"
        )

    print("=" * 130)

    # Summary statistics
    tspars = [r["tsparse_epoch"] for r in results if r["tsparse_epoch"] is not None and r.get("label", "").startswith("seed")]
    spans = [r["onset_to_frozen_span"] for r in results if r["onset_to_frozen_span"] is not None and r.get("label", "").startswith("seed")]
    growths = [r["growth_rate"] for r in results if r["growth_rate"] is not None and r.get("label", "").startswith("seed")]

    if tspars:
        print(f"\nStandard schedule (seeds 0-9):")
        print(f"  t_sparse: mean={np.mean(tspars):.1f}, std={np.std(tspars):.1f}, range=[{min(tspars)}, {max(tspars)}]")
    if spans:
        print(f"  onset→frozen span: mean={np.mean(spans):.1f}, std={np.std(spans):.1f}, range=[{min(spans)}, {max(spans)}]")
    if growths:
        print(f"  growth rate: mean={np.mean(growths):.4f}, std={np.std(growths):.4f}")

    # Basis selection distribution
    basis_counts = {}
    for r in results:
        if r.get("label", "").startswith("seed"):
            b = r["dominant_basis"]
            basis_counts[b] = basis_counts.get(b, 0) + 1
    if basis_counts:
        print(f"  basis selection: {basis_counts}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="t_sparse sweep across seeds")
    parser.add_argument("--workers", type=int, default=1,
                        help="Max parallel training processes (default 1 for GPU safety)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip training, just analyze existing checkpoints")
    parser.add_argument("--seeds", type=str, default="0-9",
                        help="Seed range, e.g. '0-9' or '0,1,5,7'")
    args = parser.parse_args()

    # Parse seed range
    if "-" in args.seeds and "," not in args.seeds:
        lo, hi = args.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    # Define runs
    runs = []
    for seed in seeds:
        runs.append({
            "seed": seed,
            "save_path": f"sweep_seed{seed}_mal.pt",
            "n_epochs": 200,
            "warmup_epochs": 50,
            "tau_final": 0.05,
            "label": f"seed={seed}",
        })

    # Slow anneal run: seed=0, 300 epochs, tau→0.001
    runs.append({
        "seed": 0,
        "save_path": "sweep_slow_seed0_mal.pt",
        "n_epochs": 300,
        "warmup_epochs": 50,
        "tau_final": 0.001,
        "label": "SLOW (seed=0)",
    })

    if not args.analyze_only:
        print(f"Running {len(runs)} training jobs with {args.workers} worker(s)...")
        print(f"Seeds: {seeds}")
        print(f"Plus 1 slow-anneal run (300 epochs, tau_final=0.001)")
        print()

        if args.workers == 1:
            # Sequential execution
            for i, run in enumerate(runs):
                print(f"[{i+1}/{len(runs)}] Training {run['label']} -> {run['save_path']}")
                seed, path, rc, elapsed, stdout, stderr = run_one_seed(
                    run["seed"], run["save_path"],
                    run["n_epochs"], run["warmup_epochs"], run["tau_final"],
                )
                status = "OK" if rc == 0 else f"FAIL (rc={rc})"
                print(f"  {status} in {elapsed:.1f}s")
                if rc != 0:
                    print(f"  STDERR: {stderr[-500:]}")
        else:
            # Parallel execution with bounded concurrency
            from concurrent.futures import ProcessPoolExecutor, as_completed

            futures = {}
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for run in runs:
                    f = pool.submit(
                        run_one_seed,
                        run["seed"], run["save_path"],
                        run["n_epochs"], run["warmup_epochs"], run["tau_final"],
                    )
                    futures[f] = run

                for f in as_completed(futures):
                    run = futures[f]
                    seed, path, rc, elapsed, stdout, stderr = f.result()
                    status = "OK" if rc == 0 else f"FAIL (rc={rc})"
                    print(f"  {run['label']:.<30s} {status} in {elapsed:.1f}s")
                    if rc != 0:
                        print(f"    STDERR: {stderr[-500:]}")

    # Analyze all checkpoints
    print("\nAnalyzing checkpoints...")
    results = []
    for run in runs:
        path = run["save_path"]
        if not os.path.exists(path):
            print(f"  SKIP {path} (not found)")
            continue
        try:
            info = detect_tsparse(path)
            info["label"] = run["label"]
            results.append(info)
        except Exception as e:
            print(f"  ERROR analyzing {path}: {e}")

    if not results:
        print("No checkpoints found. Run without --analyze-only first.")
        return

    tabulate_results(results)

    # Test user's predictions
    slow = [r for r in results if r["label"].startswith("SLOW")]
    standard = [r for r in results if r["label"].startswith("seed")]

    if slow and standard:
        slow_r = slow[0]
        std_seed0 = [r for r in standard if r["label"] == "seed=0"]

        print("\n" + "=" * 70)
        print("PREDICTION VERIFICATION")
        print("=" * 70)

        if std_seed0:
            s0 = std_seed0[0]
            print(f"\n1. t_sparse shifts right (delayed)?")
            if slow_r["tsparse_epoch"] and s0["tsparse_epoch"]:
                delta = slow_r["tsparse_epoch"] - s0["tsparse_epoch"]
                print(f"   Standard seed=0: t_sparse = {s0['tsparse_epoch']}")
                print(f"   Slow     seed=0: t_sparse = {slow_r['tsparse_epoch']}")
                print(f"   Delta = {delta:+d} epochs  -> {'YES, shifted right' if delta > 0 else 'NO'}")

        if standard:
            mean_span = np.mean([r["onset_to_frozen_span"] for r in standard if r["onset_to_frozen_span"]])
            std_span = np.std([r["onset_to_frozen_span"] for r in standard if r["onset_to_frozen_span"]])
            print(f"\n2. Onset->frozen span stays ~30 epochs?")
            print(f"   Standard: mean={mean_span:.1f} +/- {std_span:.1f}")
            if slow_r["onset_to_frozen_span"]:
                print(f"   Slow:     span={slow_r['onset_to_frozen_span']}")
                print(f"   -> {'YES, ~same' if abs(slow_r['onset_to_frozen_span'] - mean_span) < 2 * std_span + 5 else 'DIFFERENT'}")

        if standard:
            mean_gr = np.mean([r["growth_rate"] for r in standard if r["growth_rate"]])
            std_gr = np.std([r["growth_rate"] for r in standard if r["growth_rate"]])
            print(f"\n3. Growth rate stays ~1.17x?")
            print(f"   Standard: mean={mean_gr:.4f} +/- {std_gr:.4f}")
            if slow_r["growth_rate"]:
                print(f"   Slow:     rate={slow_r['growth_rate']:.4f}")
                print(f"   -> {'YES, ~same' if abs(slow_r['growth_rate'] - mean_gr) < 2 * std_gr + 0.02 else 'DIFFERENT'}")

    # Save results JSON for downstream use
    out_json = "tsparse_sweep_results.json"
    serializable = []
    for r in results:
        sr = {k: v for k, v in r.items()}
        sr["growth_rate"] = float(sr["growth_rate"]) if sr["growth_rate"] is not None else None
        serializable.append(sr)
    with open(out_json, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved results to {out_json}")


if __name__ == "__main__":
    main()
