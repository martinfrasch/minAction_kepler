# MinAction.Net: Discovering Physical Laws from Noisy Observations

**Reproducible code for:** *From Physiology to Physics: Minimum-Action Learning as a Vertically Organizing Principle for AI, Brains, and Physical Law*

MinAction.Net discovers symbolic force laws from noisy trajectory data using a **Triple-Action** optimization framework inspired by biological energy-minimization principles.

## The Triple-Action Framework

$$\mathcal{L} = \alpha_I \mathcal{L}_{I_{\max}} + \alpha_E \mathcal{L}_{E_{\min}} + \alpha_S \mathcal{L}_{\mathrm{Symmetry}}$$

| Component | Role |
|-----------|------|
| **I_max** (Information) | Trajectory reconstruction + wide-stencil acceleration matching |
| **E_min** (Energy) | L1 sparsity + gate entropy + temperature annealing |
| **L_Symmetry** (Noetherian) | SO(2) invariance + energy conservation enforcement |

The architecture uses a learnable radial basis library `[1/r², 1/r, r, 1, 1/r³]` with softmax gates that crystallize from uniform to one-hot during training via Bimodal Glial-Neural Optimization (BGNO).

## Results Summary

| Benchmark | Force Law | Direct Selection | + Noetherian | Coefficient | Period Law |
|-----------|-----------|-----------------|--------------|-------------|------------|
| Kepler | F = GM/r² | 4/10 (10/10 with biased init) | **10/10** | GM = 0.94 (6% error) | T² ∝ a³·⁰¹ |
| Hooke | F = kr | 9/10 | **10/10** | k = 0.980 (2% error) | T = const |

**Key insight:** Wide-stencil preprocessing (s=10) is the critical enabler for all methods tested, including SINDy. MAL's distinct contribution is integrated Noetherian validation — the symmetry term is intrinsic to the trifunctional, not a post-hoc check.

## Setup

```bash
conda create -n minaction python=3.10
conda activate minaction
pip install -r requirements.txt
```

Tested with PyTorch >= 2.0 + CUDA 11.8 on NVIDIA RTX 2080 Ti.

## Reproducing Paper Results

```bash
# Full reproduction of all paper experiments
python reproduce_paper.py

# Or run individually:
python main.py                                    # Kepler discovery pipeline
python train_hooke.py                             # Hooke benchmark
python run_tsparse_sweep.py                       # 10-seed Kepler sweep
python run_hooke_sweep.py                         # 10-seed Hooke sweep
python baseline_sindy_robust.py --seeds 0-9       # SINDy variants
python baseline_hnn.py                            # Hamiltonian NN
python baseline_lnn.py                            # Lagrangian NN
python analyze_basis_selection.py --interventions  # Basis selection analysis
python run_biased_init.py                         # Physics-informed initialization
```

## Project Structure

```
minaction_model.py              Core model: NoetherForceBasis, MinActionNet, Triple-Action loss
data_kepler.py                  Kepler orbit data generation (16 orbits, symplectic integrator)
data_hooke.py                   Hooke harmonic oscillator data generation
train_mal_with_metrics.py       Full Kepler training (200 epochs, BGNO schedule)
train_hooke.py                  Hooke training with full metrics
train_minaction.py              Minimal Kepler training (50 epochs)
evaluate_minaction.py           Evaluation and visualization
baseline_sindy_robust.py        SINDy, GP-SINDy, Ensemble-SINDy comparison
baseline_hnn.py                 Hamiltonian Neural Network baseline
baseline_lnn.py                 Lagrangian Neural Network baseline
run_tsparse_sweep.py            10-seed Kepler robustness sweep
run_hooke_sweep.py              10-seed Hooke robustness sweep
analyze_basis_selection.py      Basis selection diagnostics + interventions
run_biased_init.py              Physics-informed gate initialization (10/10)
reproduce_paper.py              Reproduce all paper experiments end-to-end
tsparse_sweep_results.json      Pre-computed 10-seed sweep results
```

## Key Design Decisions

1. **Triple-Action functional:** Three objectives jointly constrain model selection — information extraction, metabolic cost minimization, and Noetherian symmetry enforcement.
2. **Soft-to-discrete manifold:** BGNO drives gates from uniform (soft) to one-hot (discrete) via temperature annealing, implementing energy-driven architectural crystallization.
3. **Wide-stencil derivatives (stride s=10):** Noise variance drops by s⁴ = 10,000×, transforming SNR from 0.02 to 1.6.
4. **Intrinsic Noetherian selection:** Energy conservation is built into the trifunctional via L_Symmetry; applying this criterion across seeds achieves 100% pipeline accuracy.
5. **Post-training calibration:** Least-squares projection recovers true force magnitude after L1-biased training.

## Citation

```bibtex
@article{Frasch2026MAL,
  author = {Frasch, Martin G.},
  title = {From Physiology to Physics: Minimum-Action Learning as a Vertically
           Organizing Principle for AI, Brains, and Physical Law},
  journal = {TBD},
  year = {2026},
  note = {Under review}
}
```

## License

MIT License — see [LICENSE](LICENSE).
