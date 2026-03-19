# Minimum-Action Learning (MAL)

**Reproducible code for:** Frasch, M. G. (2026). *Minimum-Action Learning: Energy-Constrained Symbolic Model Selection for Physical Law Identification from Noisy Data.* arXiv:2603.16951. [https://arxiv.org/abs/2603.16951](https://arxiv.org/abs/2603.16951)

MAL discovers symbolic force laws from noisy trajectory data using a **Triple-Action** optimization framework that combines information maximization, energy-constrained sparsity, and Noetherian symmetry enforcement.

## The Triple-Action Framework

$$\mathcal{L} = \alpha_I \mathcal{L}_{I_{\max}} + \alpha_E \mathcal{L}_{E_{\min}} + \alpha_S \mathcal{L}_{\mathrm{Symmetry}}$$

| Component | Role |
|-----------|------|
| **I_max** (Information) | Trajectory reconstruction + wide-stencil acceleration matching |
| **E_min** (Energy) | L1 sparsity + gate entropy + temperature annealing |
| **L_Symmetry** (Noetherian) | SO(2) invariance + energy conservation enforcement |

The architecture uses a learnable radial basis library `[1/r², 1/r, r, 1, 1/r³]` with softmax gates that crystallize from uniform to one-hot during training via Bimodal Glial-Neural Optimization (BGNO). A wide-stencil preprocessing step (stride s=10) reduces noise variance by s⁴ = 10,000×, transforming an intractable estimation problem (SNR ≈ 0.02) into a learnable one (SNR ≈ 1.6).

## Results Summary

| Benchmark | Force Law | Direct Selection | + Noetherian | Coefficient | Period Law |
|-----------|-----------|-----------------|--------------|-------------|------------|
| Kepler | F = GM/r² | 4/10 (10/10 with biased init) | **10/10** | GM = 0.94 (6% error) | T² ∝ a³·⁰¹ |
| Hooke | F = kr | 9/10 | **10/10** | k = 0.980 (2% error) | T = const |

The 40% raw Kepler rate is a feature of the pipeline: incorrect bases match short-term trajectories but violate energy conservation by 3–6×, providing a physics-grounded filter (Noetherian model selection) that achieves 100% identification.

### Baseline Comparisons

| Method | Basis ID | Interpretable | Dynamical Validation | Energy |
|--------|----------|---------------|---------------------|--------|
| **MAL (ours)** | **10/10** (Noetherian) | Yes (symbolic) | Rollout + conservation | 0.07 kWh |
| Vanilla SINDy (s=10) | 10/10 | Yes | No | <0.001 kWh |
| GP-SINDy | 8/10 | Yes | No | <0.001 kWh |
| Ensemble-SINDy | 10/10 | Yes | No | <0.001 kWh |
| HNN | N/A (black-box) | No | By construction | 0.006 kWh |
| LNN | N/A (failed) | No | Hessian singular | 0.013 kWh |

Wide-stencil preprocessing is the critical enabler shared by all methods. SINDy achieves comparable basis selection at <1% of MAL's cost. MAL's distinct contribution is **integrated dynamical validation** — trajectory rollout + Noetherian energy conservation — that SINDy lacks.

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
python run_basis_sensitivity.py                   # Basis library sensitivity (variable K)
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
run_basis_sensitivity.py        Basis library sensitivity (confounders, missing basis, expanded)
reproduce_paper.py              Reproduce all paper experiments end-to-end
```

## Key Design Decisions

1. **Triple-Action functional:** Three objectives jointly constrain model selection — information extraction, metabolic cost minimization, and Noetherian symmetry enforcement.
2. **Soft-to-discrete manifold:** BGNO drives gates from uniform (soft) to one-hot (discrete) via temperature annealing, implementing energy-driven architectural crystallization.
3. **Wide-stencil derivatives (stride s=10):** Noise variance drops by s⁴ = 10,000×, transforming SNR from 0.02 to 1.6.
4. **Noetherian model selection:** Energy conservation discriminates correct from incorrect bases — incorrect bases violate conservation by 3–6×, providing a physics-grounded filter that achieves 100% pipeline accuracy.
5. **Post-training calibration:** Least-squares projection recovers true force magnitude after L1-biased training.

## Citation

```bibtex
@article{Frasch2026MAL,
  author  = {Frasch, Martin G.},
  title   = {Minimum-Action Learning: Energy-Constrained Symbolic Model
             Selection for Physical Law Identification from Noisy Data},
  year    = {2026},
  eprint  = {2603.16951},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG},
  url     = {https://arxiv.org/abs/2603.16951}
}
```

[https://doi.org/10.5281/zenodo.18918528](https://doi.org/10.5281/zenodo.18918528)

## License

MIT License — see [LICENSE](LICENSE).
