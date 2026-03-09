"""
MAL (Minimum-Action Learning) trainer with weight-path action tracking.

Mirrors the original train_minaction.py training loop but adds per-epoch:
  - Weight-path action: sum_steps ||Δw||²  (parameter-space kinetic energy)
  - Wall-clock time
  - Energy proxy: num_params * num_steps
  - Full loss component breakdown

Saves all metrics alongside the model checkpoint for comparison with
the MSE-only baseline (train_mse_with_metrics.py).
"""

import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data_kepler import OrbitConfig, generate_dataset, train_val_test_split
from minaction_model import MinActionNet, minaction_loss, calibrate_theta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Re-use the same data pipeline from train_minaction.py
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
# Weight-path action helpers
# ---------------------------------------------------------------------------
def flatten_params(model):
    """Concatenate all parameters into a single 1-D CPU vector."""
    return torch.cat([p.detach().cpu().view(-1) for p in model.parameters()])


def compute_step_action(prev_w, curr_w):
    """Discrete ||Δw||² — one increment of the parameter-path action."""
    delta = curr_w - prev_w
    return float(torch.dot(delta, delta))


# ---------------------------------------------------------------------------
# Period estimation & Kepler exponent (same as train_minaction.py)
# ---------------------------------------------------------------------------
def estimate_period_from_traj(r_clean, t):
    x = r_clean[:, 0] - np.mean(r_clean[:, 0])
    N = len(x)
    corr = np.correlate(x, x, mode="full")
    corr = corr[N - 1:]
    corr = corr / np.arange(N, 0, -1)
    corr = corr / corr[0]
    i = 1
    while i < N - 1 and corr[i] <= corr[i - 1]:
        i += 1
    while i < N - 1 and corr[i] <= corr[i + 1]:
        i += 1
    return t[i] - t[0]


def fit_kepler_exponent(a_vals, T_vals):
    y = np.log(T_vals ** 2)
    x = np.log(a_vals)
    A = np.vstack([x, np.ones_like(x)]).T
    p_hat, logC_hat = np.linalg.lstsq(A, y, rcond=None)[0]
    return p_hat, np.exp(logC_hat)


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------
def main(save_path="mal_with_metrics.pt", data_splits=None, seed=42,
         n_epochs=200, warmup_epochs=50, tau_final=0.05, quiet=False,
         A_logits_init=None):
    """
    Train MAL model with full metrics tracking.

    Args:
        save_path: checkpoint output path
        data_splits: optional (train_data, val_data, test_data) tuple.
                     If None, generates fresh data using seed.
        seed: random seed for data generation and model init (ignored if
              data_splits is provided for data, still used for model init).
        n_epochs: total training epochs
        warmup_epochs: epochs with tau=1.0 before annealing begins
        tau_final: final temperature for gate softmax
        quiet: if True, suppress per-epoch printing
    """
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    if data_splits is not None:
        train_data, val_data, test_data = data_splits
    else:
        rng = np.random.default_rng(seed)
        cfg = OrbitConfig()
        sigma = 1e-2
        data = generate_dataset(cfg, sigma=sigma, rng=rng)
        train_data, val_data, test_data = train_val_test_split(data, rng=rng)

    max_T = 200
    train_ds = OrbitDataset(train_data, max_T=max_T)
    val_ds = OrbitDataset(val_data, max_T=max_T)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)

    dt_obs_example = train_data["t"][0][1] - train_data["t"][0][0]
    model_dt = dt_obs_example / 5.0
    model = MinActionNet(dt=model_dt).to(device)
    if A_logits_init is not None:
        with torch.no_grad():
            model.force_basis.A_logits.copy_(torch.tensor(A_logits_init, dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not quiet:
        print(f"[MAL] Trainable parameters: {num_params}")
        print(f"[MAL] Device: {device}")

    print_every = 10

    # Metric accumulators
    epoch_train_loss = []
    epoch_val_loss = []
    epoch_weight_action = []
    epoch_wall_time = []
    epoch_energy_proxy = []
    epoch_loss_components = []   # list of dicts per epoch
    epoch_gates = []
    epoch_theta = []

    prev_w = flatten_params(model)
    start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        # Two-phase schedule: warmup then sparsify
        if epoch <= warmup_epochs:
            alpha_E = 0.01
            tau = 1.0
        else:
            progress = (epoch - warmup_epochs) / (n_epochs - warmup_epochs)
            alpha_E = 0.01 + 0.99 * progress
            tau = 1.0 * (tau_final / 1.0) ** progress
        model.force_basis.tau = tau

        # --- Train ---
        model.train()
        epoch_start = time.time()
        train_loss_accum = 0.0
        last_loss_dict = None
        running_action = 0.0
        step_count = 0

        for r_obs, dt_obs in train_loader:
            r_obs = r_obs.to(device)
            dt_obs = dt_obs.to(device)
            optimizer.zero_grad()
            loss, loss_dict = minaction_loss(model, r_obs, dt_obs.item(), alpha_E=alpha_E)
            loss.backward()
            optimizer.step()
            train_loss_accum += loss.item() * r_obs.size(0)
            last_loss_dict = loss_dict
            step_count += 1

            # Weight-path action
            curr_w = flatten_params(model)
            running_action += compute_step_action(prev_w, curr_w)
            prev_w = curr_w

        train_loss = train_loss_accum / len(train_ds)
        epoch_time = time.time() - epoch_start

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            val_loss_accum = 0.0
            for r_obs, dt_obs in val_loader:
                r_obs = r_obs.to(device)
                dt_obs = dt_obs.to(device)
                loss, _ = minaction_loss(model, r_obs, dt_obs.item(), alpha_E=alpha_E)
                val_loss_accum += loss.item() * r_obs.size(0)
            val_loss = val_loss_accum / len(val_ds)

        # Record metrics
        epoch_train_loss.append(train_loss)
        epoch_val_loss.append(val_loss)
        epoch_weight_action.append(running_action)
        epoch_wall_time.append(epoch_time)
        epoch_energy_proxy.append(num_params * step_count)
        epoch_loss_components.append(last_loss_dict)

        with torch.no_grad():
            gates = torch.softmax(model.force_basis.A_logits / tau, dim=0).cpu().tolist()
            theta = model.force_basis.theta.cpu().tolist()
        epoch_gates.append(gates)
        epoch_theta.append(theta)

        if not quiet and (epoch % print_every == 0 or epoch == 1):
            elapsed = time.time() - start_time
            print(
                f"[MAL] Epoch {epoch:03d} | train={train_loss:.4e} | val={val_loss:.4e} | "
                f"action={running_action:.3e} | alpha_E={alpha_E:.3f} | tau={tau:.3f} | "
                f"time={elapsed:.1f}s"
            )
            if last_loss_dict:
                print(
                    f"  L_traj={last_loss_dict['L_traj']:.4e}  L_accel={last_loss_dict['L_accel']:.4e}  "
                    f"L_comp={last_loss_dict['L_comp']:.4e}  L_arch={last_loss_dict['L_arch']:.4e}"
                )
            print(f"  Gates A: {[f'{g:.4f}' for g in gates]}")
            print(f"  Theta:   {[f'{t:.4f}' for t in theta]}")

    total_time = time.time() - start_time
    if not quiet:
        print(f"[MAL] Training finished in {total_time:.1f}s")

    # Post-training calibration
    dt_obs_cal = train_data["t"][0][1] - train_data["t"][0][0]
    calibrate_theta(model, train_data, dt_obs_cal)
    with torch.no_grad():
        gates_final = torch.softmax(model.force_basis.A_logits / model.force_basis.tau, dim=0).cpu().tolist()
        theta_final = model.force_basis.theta.cpu().tolist()
    if not quiet:
        print(f"  Calibrated theta[{np.argmax(gates_final)}] = {theta_final[np.argmax(gates_final)]:.4f}")
        print(f"  Final theta: {[f'{t:.4f}' for t in theta_final]}")
        print(f"  Final gates: {[f'{g:.4f}' for g in gates_final]}")

    # Energy estimate
    assumed_power_W = 200.0
    energy_kWh = assumed_power_W * total_time / 3.6e6
    if not quiet:
        print(f"[MAL] Approx. training energy: {energy_kWh:.6f} kWh (assuming {assumed_power_W}W)")

    # Kepler exponent
    a_vals = test_data["a"]
    T_clean = []
    for r_clean, t in zip(test_data["r_clean"], test_data["t"]):
        T_est = estimate_period_from_traj(np.array(r_clean), np.array(t))
        T_clean.append(T_est)
    T_clean = np.array(T_clean)
    p_hat, C_hat = fit_kepler_exponent(a_vals, T_clean)
    if not quiet:
        print(f"[MAL] Fitted Kepler exponent p_hat = {p_hat:.4f} (expected ~3.0)")

    # Save everything
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_dt": model_dt,
        "epoch_train_loss": epoch_train_loss,
        "epoch_val_loss": epoch_val_loss,
        "epoch_weight_action": epoch_weight_action,
        "epoch_wall_time": epoch_wall_time,
        "epoch_energy_proxy": epoch_energy_proxy,
        "epoch_loss_components": epoch_loss_components,
        "epoch_gates": epoch_gates,
        "epoch_theta": epoch_theta,
        "num_params": num_params,
        "total_time": total_time,
        "energy_kWh": energy_kWh,
        "kepler_p_hat": p_hat,
        "kepler_C_hat": C_hat,
        "test_data": test_data,
        "schedule": {
            "n_epochs": n_epochs,
            "warmup_epochs": warmup_epochs,
            "tau_final": tau_final,
        },
    }, save_path)
    if not quiet:
        print(f"[MAL] Saved model + metrics to {save_path}")

    return model, energy_kWh, test_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="mal_with_metrics.pt")
    args = parser.parse_args()
    main(save_path=args.save, seed=args.seed)
