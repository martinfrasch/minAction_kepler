#!/usr/bin/env python
"""
Reproduce all experiments from the MinAction.Net paper.

Runs sequentially:
  1. Kepler 10-seed robustness sweep (MAL)
  2. Hooke's law 10-seed sweep
  3. SINDy comparison (vanilla, GP, ensemble) across 10 seeds
  4. HNN baseline
  5. LNN baseline
  6. Basis selection analysis with interventions

Outputs: JSON result files + console summary.
Estimated runtime: ~6 hours on a single GPU (RTX 2080 Ti).

Usage:
    python reproduce_paper.py           # run everything
    python reproduce_paper.py --skip-training  # just analyze existing checkpoints
"""

import subprocess
import sys
import os
import time

SCRIPTS = [
    ("Kepler 10-seed sweep", "run_tsparse_sweep.py", []),
    ("Hooke 10-seed sweep", "run_hooke_sweep.py", []),
    ("SINDy comparison (10 seeds)", "baseline_sindy_robust.py", ["--seeds", "0-9"]),
    ("HNN baseline", "baseline_hnn.py", []),
    ("LNN baseline", "baseline_lnn.py", []),
    ("Basis selection analysis", "analyze_basis_selection.py", ["--interventions"]),
    ("Basis library sensitivity", "run_basis_sensitivity.py", []),
]

ANALYZE_ONLY = [
    ("Kepler sweep (analyze)", "run_tsparse_sweep.py", ["--analyze-only"]),
    ("Hooke sweep (analyze)", "run_hooke_sweep.py", ["--analyze-only"]),
    ("Basis selection (analyze)", "analyze_basis_selection.py", []),
]


def run_script(name, script, args, cwd):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script] + args,
        cwd=cwd,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"\n  [{status}] {name} — {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return result.returncode == 0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, just analyze existing checkpoints")
    args = parser.parse_args()

    cwd = os.path.dirname(os.path.abspath(__file__))
    scripts = ANALYZE_ONLY if args.skip_training else SCRIPTS

    print("=" * 60)
    print("  MinAction.Net — Reproduce All Paper Experiments")
    print("=" * 60)
    print(f"  Mode: {'analyze only' if args.skip_training else 'full training'}")
    print(f"  Working directory: {cwd}")
    print(f"  Scripts to run: {len(scripts)}")

    t_total = time.time()
    results = []
    for name, script, script_args in scripts:
        ok = run_script(name, script, script_args, cwd)
        results.append((name, ok))

    total_time = time.time() - t_total
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL':>4s}  {name}")
    print(f"\n  Total time: {total_time:.0f}s ({total_time/3600:.1f} hours)")

    # List output files
    print(f"\n  Output files:")
    for f in sorted(os.listdir(cwd)):
        if f.endswith(".json"):
            size = os.path.getsize(os.path.join(cwd, f))
            print(f"    {f} ({size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
