#!/usr/bin/env python
"""
Run the biased_init intervention: A_logits[0] = 1.5 (give 1/r^2 a head start).
This properly passes A_logits_init to train_mal_with_metrics.main().
"""
import os, sys, time, json, subprocess
import numpy as np

BASIS_LABELS = ["1/r^2", "1/r", "r", "1", "1/r^3"]

def main():
    seeds = list(range(10))
    results = []

    print("Running biased_init intervention (10 seeds)...")
    for seed in seeds:
        save_path = f"biased_init_seed{seed}.pt"
        script = f"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
from data_kepler import OrbitConfig, generate_dataset, train_val_test_split
from train_mal_with_metrics import main as train_mal

seed = {seed}
rng_data = np.random.default_rng(seed)
cfg = OrbitConfig()
data = generate_dataset(cfg, sigma=1e-2, rng=rng_data)
rng_split = np.random.default_rng(seed)
train_data, val_data, test_data = train_val_test_split(data, rng=rng_split)

train_mal(
    save_path="{save_path}",
    data_splits=(train_data, val_data, test_data),
    seed=seed,
    n_epochs=200,
    warmup_epochs=50,
    tau_final=0.05,
    quiet=True,
    A_logits_init=[1.5, 0.0, 0.0, 0.0, 0.0],
)
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
        print(f"  [{seed+1}/10] seed={seed}: {status} ({elapsed:.0f}s)")
        if result.returncode != 0:
            print(f"    {result.stderr[-300:]}")

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
            os.remove(save_path)

    n_correct = sum(1 for r in results if r["selected_1_r2"])
    print(f"\nBiased init result: {n_correct}/{len(results)} select 1/r^2")
    for r in results:
        print(f"  seed={r['seed']}: {r['dominant_basis']} (p_hat={r['kepler_p_hat']:.4f})")

    with open("biased_init_results.json", "w") as f:
        json.dump({"name": "biased_init", "n_correct": n_correct,
                    "n_total": len(results), "results": results}, f, indent=2)
    print("Saved to biased_init_results.json")


if __name__ == "__main__":
    main()
