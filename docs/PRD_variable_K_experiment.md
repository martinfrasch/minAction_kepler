# PRD: Variable-K Basis Sensitivity Experiment

**Status**: Ready for implementation
**Target**: SI Section S4F, Table S6
**Date**: 2026-03-24

---

## 1. Objective

Demonstrate that the MinAction.Net pipeline reliably selects the correct force law (1/r^2) across varying basis library sizes and compositions, including when confounders are added and when the correct basis is absent. This experiment produces SI Table S6 and the supporting paragraph in Section S4F.

## 2. Background

The standard pipeline uses a fixed K=5 basis library {1/r^2, 1/r, r, 1, 1/r^3}. Reviewers and readers will reasonably ask:

- **Does adding confounders hurt selection?** If we pad the library with plausible but incorrect terms, does the gate still concentrate on 1/r^2?
- **Does a larger search space degrade performance?** With K=8, does sparsification take longer or fail?
- **What happens when the correct answer is missing?** The pipeline should fail gracefully (no single dominant gate, poor energy conservation) rather than silently picking a wrong answer.

These questions are critical for establishing that the Noetherian gate-and-sparsify mechanism is robust, not fragile to the specific library used during the original 10-seed sweep.

## 3. Code Changes Required

### 3.1 Modify `minaction_model.py` — `NoetherForceBasis.__init__` and `get_phi`

**Current state**: `get_phi()` has a hardcoded 5-term basis library (lines 20-28). `n_terms` only truncates the list.

**Required change**: Accept a configurable list of basis functions.

```python
class NoetherForceBasis(nn.Module):
    def __init__(self, n_terms=5, basis_fns=None):
        super().__init__()
        if basis_fns is not None:
            self.basis_fns = basis_fns
            self.n_terms = len(basis_fns)
        else:
            # Default: original 5-term library for backward compatibility
            self.basis_fns = None
            self.n_terms = n_terms
        self.theta = nn.Parameter(torch.randn(self.n_terms) * 0.1)
        self.A_logits = nn.Parameter(torch.ones(self.n_terms))
        self.tau = 1.0

    def get_phi(self, r):
        eps = 1e-8
        if self.basis_fns is not None:
            # Custom basis: each fn takes r (tensor) and returns a scalar field
            phi = [fn(r, eps) for fn in self.basis_fns]
            return torch.cat(phi, dim=-1)
        else:
            # Original hardcoded basis (backward compatible)
            phi = []
            phi.append(1.0 / (r**2 + eps))
            phi.append(1.0 / (r + eps))
            phi.append(r)
            phi.append(torch.ones_like(r))
            phi.append(1.0 / (r**3 + eps))
            return torch.cat(phi[:self.n_terms], dim=-1)
```

**Backward compatibility**: When `basis_fns=None` (the default), behavior is identical to the current code. No existing scripts break.

### 3.2 Propagate `basis_fns` through `MinActionNet`

```python
class MinActionNet(nn.Module):
    def __init__(self, dt, n_terms=5, basis_fns=None):
        super().__init__()
        self.dt = dt
        self.force_basis = NoetherForceBasis(n_terms=n_terms, basis_fns=basis_fns)
```

### 3.3 Adapt `calibrate_theta()`

No structural changes needed. It already operates on `model.force_basis.get_phi(r_mag)` and indexes by `dominant`. It will work with any K as long as `get_phi` returns the right shape.

### 3.4 Adapt `minaction_loss()`

No structural changes needed. The loss function reads `model.force_basis.A_logits` and `model.force_basis.theta` directly, which will have the correct size from `__init__`.

### 3.5 Adapt `train_mal_with_metrics.py`

Add `basis_fns` parameter to `main()` and pass it through to `MinActionNet`:

```python
def main(save_path=..., ..., basis_fns=None):
    ...
    model = MinActionNet(dt=model_dt, basis_fns=basis_fns).to(device)
```

**One line change** at model construction (currently line 138). The rest of the training loop is generic.

### 3.6 Create `run_basis_sensitivity.py` (new file)

See Section 4 for full experiment design. This script follows the pattern of `run_tsparse_sweep.py`:
- Define experiment conditions as dicts
- Run training via subprocess workers (one GPU at a time)
- Analyze checkpoints and produce summary table + JSON output

## 4. Experiment Design

### 4.1 Conditions

| Condition | K | Basis Functions | Labels | Rationale |
|-----------|---|----------------|--------|-----------|
| **Standard** | 5 | {1/r^2, 1/r, r, 1, 1/r^3} | `["1/r^2","1/r","r","1","1/r^3"]` | Control — matches existing 10-seed sweep |
| **Confounders** | 7 | Standard + {r^-2.5, r^-1.5} | `["1/r^2","1/r","r","1","1/r^3","1/r^2.5","1/r^1.5"]` | Near-neighbor confounders close to 1/r^2 |
| **Expanded** | 8 | Standard + {r^2, 1/r^4, ln(r)} | `["1/r^2","1/r","r","1","1/r^3","r^2","1/r^4","ln(r)"]` | Larger search space with diverse terms |
| **Missing** | 4 | {1/r, r, 1, 1/r^3} | `["1/r","r","1","1/r^3"]` | Correct basis absent — failure mode test |

### 4.2 Basis Function Definitions

```python
EXPERIMENTS = {
    "standard": {
        "K": 5,
        "labels": ["1/r^2", "1/r", "r", "1", "1/r^3"],
        "basis_fns": [
            lambda r, eps: 1.0 / (r**2 + eps),
            lambda r, eps: 1.0 / (r + eps),
            lambda r, eps: r,
            lambda r, eps: torch.ones_like(r),
            lambda r, eps: 1.0 / (r**3 + eps),
        ],
    },
    "confounders": {
        "K": 7,
        "labels": ["1/r^2", "1/r", "r", "1", "1/r^3", "1/r^2.5", "1/r^1.5"],
        "basis_fns": [
            # ... standard 5 ...
            lambda r, eps: 1.0 / (r**2.5 + eps),
            lambda r, eps: 1.0 / (r**1.5 + eps),
        ],
    },
    "expanded": {
        "K": 8,
        "labels": ["1/r^2", "1/r", "r", "1", "1/r^3", "r^2", "1/r^4", "ln(r)"],
        "basis_fns": [
            # ... standard 5 ...
            lambda r, eps: r**2,
            lambda r, eps: 1.0 / (r**4 + eps),
            lambda r, eps: torch.log(r + eps),
        ],
    },
    "missing": {
        "K": 4,
        "labels": ["1/r", "r", "1", "1/r^3"],
        "basis_fns": [
            lambda r, eps: 1.0 / (r + eps),
            lambda r, eps: r,
            lambda r, eps: torch.ones_like(r),
            lambda r, eps: 1.0 / (r**3 + eps),
        ],
    },
}
```

### 4.3 Per-Condition Protocol

For each condition, run seeds 0-9 (10 seeds). Each seed:

1. Generate Kepler orbit data (same `OrbitConfig`, sigma=0.01)
2. Train MinActionNet with the condition's basis library (200 epochs, warmup=50, tau_final=0.05)
3. Run `calibrate_theta()` post-training
4. Record metrics (see Section 5)

### 4.4 Subprocess Architecture

Follow `run_tsparse_sweep.py` pattern: inline Python script passed via `subprocess.run([sys.executable, "-c", script])`. The worker script must:

1. Import the basis function definitions (cannot pickle lambdas across processes)
2. Construct `basis_fns` list inside the subprocess
3. Pass `basis_fns` to `train_mal_with_metrics.main()`

**Implementation note**: Since lambdas cannot be serialized, define basis functions as module-level named functions or reconstruct them in the worker script string. The cleanest approach: define a `make_basis_fns(condition_name)` factory function and call it inside the worker.

## 5. Expected Outputs

### 5.1 Checkpoint Files

```
basis_sensitivity/
    standard_seed0_mal.pt
    standard_seed1_mal.pt
    ...
    confounders_seed0_mal.pt
    ...
    expanded_seed0_mal.pt
    ...
    missing_seed0_mal.pt
    ...
```

40 checkpoint files total, saved in a `basis_sensitivity/` subdirectory.

### 5.2 JSON Results File

`basis_sensitivity_results.json` — one entry per run:

```json
[
  {
    "condition": "standard",
    "K": 5,
    "seed": 0,
    "selected_basis": "1/r^2",
    "selected_idx": 0,
    "C_gate": 0.98,
    "sigma_H": 0.0012,
    "theta_calibrated": 1.0003,
    "training_time_s": 835.2,
    "correct": true
  },
  ...
]
```

**Field definitions**:

| Field | Description | Computation |
|-------|-------------|-------------|
| `condition` | Experiment name | From config |
| `K` | Basis library size | `len(basis_fns)` |
| `seed` | Random seed | 0-9 |
| `selected_basis` | Label of dominant gate | `labels[argmax(A)]` |
| `selected_idx` | Index of dominant gate | `argmax(softmax(A_logits / tau))` |
| `C_gate` | Gate concentration (HHI) | `sum(A_i^2)` where A = softmax(A_logits/tau) |
| `sigma_H` | Energy conservation std dev | `std(E)` over rollout positions |
| `theta_calibrated` | Calibrated coefficient of dominant basis | From `calibrate_theta()` |
| `training_time_s` | Wall-clock training time | From checkpoint |
| `correct` | Whether selected basis is 1/r^2 | `selected_basis == "1/r^2"` |

### 5.3 Summary Table (SI Table S6 format)

Printed to stdout and saved as `basis_sensitivity_table.txt`:

```
================================================================
BASIS SENSITIVITY EXPERIMENT — SI Table S6
================================================================
Condition     | K | Correct | C_gate        | sigma_H       | Time (s)
              |   | (n/10)  | mean +/- std  | mean +/- std  | mean +/- std
--------------+---+---------+---------------+---------------+----------
Standard      | 5 |  4/10   | 0.97 +/- 0.02 | 0.001 +/- ... | 835 +/- ...
Confounders   | 7 |  ?/10   | ...           | ...           | ...
Expanded      | 8 |  ?/10   | ...           | ...           | ...
Missing       | 4 |  0/10   | ...           | ...           | ...
================================================================
```

**Note on "Correct" column**: For Standard, expect ~4/10 raw rate (matching existing results). For Missing, expect 0/10 (correct answer absent). The Noetherian diagnostic (not run in this experiment) would boost Standard/Confounders/Expanded to ~10/10.

## 6. Success Criteria

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Standard matches existing | 4/10 correct (within +/-2) | Reproduces known raw selection rate |
| Confounders don't degrade | >= 3/10 correct | Near-neighbor terms shouldn't dramatically hurt |
| Expanded doesn't degrade | >= 3/10 correct | Larger K shouldn't break sparsification |
| Missing fails gracefully | 0/10 correct, low C_gate | No false positive when ground truth is absent |
| Missing has poor energy | sigma_H >> Standard sigma_H | Wrong law can't conserve energy |
| C_gate high when correct | C_gate > 0.9 for correct selections | Gate concentration validates selection |
| All runs complete | 40/40 finish without crash | No numerical instabilities from new bases |

## 7. Runtime Estimate

| Condition | Seeds | Time/seed | Subtotal |
|-----------|-------|-----------|----------|
| Standard | 10 | ~835s | 2.3 hrs |
| Confounders | 10 | ~900s (larger K) | 2.5 hrs |
| Expanded | 10 | ~950s (K=8) | 2.6 hrs |
| Missing | 10 | ~800s (K=4) | 2.2 hrs |
| **Total** | **40** | | **~9.6 hrs** |

Hardware: NVIDIA RTX 2080 Ti (11 GB VRAM). Single-GPU sequential execution (one seed at a time). Larger K increases per-epoch time slightly due to more parameters and wider softmax, but the effect is small since the basis evaluation is not the bottleneck.

## 8. Instructions for Running on CUDA Workstation

### 8.1 Prerequisites

```bash
cd ~/minAction_kepler-main          # or wherever the repo lives
conda activate minaction
git pull                            # get latest code with basis_fns support
```

### 8.2 Verify Code Changes

Before the full run, do a smoke test:

```bash
# Quick test: 1 seed, 5 epochs, standard basis
python run_basis_sensitivity.py --conditions standard --seeds 0 --epochs 5

# Verify checkpoint was created
ls basis_sensitivity/standard_seed0_mal.pt

# Quick test: 1 seed, 5 epochs, expanded basis (K=8)
python run_basis_sensitivity.py --conditions expanded --seeds 0 --epochs 5
```

### 8.3 Full Run

```bash
# Run all 4 conditions x 10 seeds (sequential, ~9.6 hours)
python run_basis_sensitivity.py

# Or run one condition at a time (useful for resuming after interruption):
python run_basis_sensitivity.py --conditions standard --seeds 0-9
python run_basis_sensitivity.py --conditions confounders --seeds 0-9
python run_basis_sensitivity.py --conditions expanded --seeds 0-9
python run_basis_sensitivity.py --conditions missing --seeds 0-9
```

### 8.4 Analyze-Only (after training completes)

```bash
# Re-generate table from existing checkpoints
python run_basis_sensitivity.py --analyze-only
```

### 8.5 Expected Output Location

```
basis_sensitivity/                    # checkpoint directory
    standard_seed0_mal.pt ... standard_seed9_mal.pt
    confounders_seed0_mal.pt ... confounders_seed9_mal.pt
    expanded_seed0_mal.pt ... expanded_seed9_mal.pt
    missing_seed0_mal.pt ... missing_seed9_mal.pt
basis_sensitivity_results.json        # machine-readable results
basis_sensitivity_table.txt           # human-readable summary table
```

### 8.6 Post-Experiment

1. Copy `basis_sensitivity_results.json` back to local machine
2. Integrate into manuscript: create SI Table S6 from the summary
3. Write 1-paragraph summary for Section S4F describing results
4. If Confounders/Expanded degrade significantly, investigate and discuss in manuscript

## 9. CLI Interface

```
usage: run_basis_sensitivity.py [-h]
                                [--conditions {standard,confounders,expanded,missing} ...]
                                [--seeds SEEDS]
                                [--epochs N]
                                [--analyze-only]
                                [--output-dir DIR]

Variable-K Basis Sensitivity Experiment (SI Table S6)

optional arguments:
  --conditions      Conditions to run (default: all four)
  --seeds           Seed range, e.g. '0-9' or '0,1,5' (default: '0-9')
  --epochs          Training epochs per seed (default: 200)
  --analyze-only    Skip training, analyze existing checkpoints
  --output-dir      Directory for checkpoints (default: 'basis_sensitivity')
```

## 10. Files Modified/Created

| File | Action | Description |
|------|--------|-------------|
| `minaction_model.py` | **Modify** | Add `basis_fns` parameter to `NoetherForceBasis` and `MinActionNet` |
| `train_mal_with_metrics.py` | **Modify** | Add `basis_fns` parameter to `main()`, pass to model constructor |
| `run_basis_sensitivity.py` | **Create** | New experiment runner script |
| `basis_sensitivity/` | **Create** (at runtime) | Checkpoint output directory |

## 11. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Lambda serialization across subprocess | High | Use string-based function reconstruction in worker script (same pattern as `run_tsparse_sweep.py`) |
| Numerical instability with ln(r) near r=0 | Medium | Use `torch.log(r + eps)` with eps=1e-8 |
| Confounders (r^-2.5) cause gradient issues | Low | Same eps protection as existing basis terms |
| Missing condition doesn't converge | Expected | This is the desired result — record it as such |
| Checkpoint compatibility with analysis | Low | Store `basis_labels` in checkpoint dict so analysis knows the mapping |

## 12. Manuscript Integration

After the experiment completes:

1. **SI Table S6**: Format results into LaTeX table matching the summary table output
2. **Section S4F paragraph**: ~150 words summarizing:
   - Standard reproduces 4/10 raw rate
   - Confounders/Expanded: gate still concentrates (C_gate > 0.9) despite larger K
   - Missing: graceful failure (low C_gate, high sigma_H, no false positive)
3. **Update Fig S6** if needed: could add a panel showing C_gate distribution across conditions
