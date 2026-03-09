"""
Hamiltonian Neural Network (HNN) baseline for the Kepler benchmark.

Learns H(q, p) as a neural network such that Hamilton's equations are satisfied:
    dq/dt = ∂H/∂p,  dp/dt = -∂H/∂q

Trained on noisy Kepler orbit data with wide-stencil velocity/acceleration estimates.

Key differences from MAL:
  - Black-box: no interpretable basis functions
  - Energy conservation by construction (symplectic structure)
  - No symbolic force law discovery

Usage:
    python baseline_hnn.py
    python baseline_hnn.py --seed 42 --epochs 200
"""

import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from data_kepler import OrbitConfig, generate_dataset, train_val_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HNN(nn.Module):
    """Hamiltonian Neural Network: learns H(q, p) and derives dynamics."""

    def __init__(self, input_dim=4, hidden=128, n_layers=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, q, p):
        x = torch.cat([q, p], dim=-1)
        return self.net(x)

    def time_derivative(self, q, p):
        """Compute Hamilton's equations: dq/dt = dH/dp, dp/dt = -dH/dq."""
        q_ = q.clone().requires_grad_(True)
        p_ = p.clone().requires_grad_(True)
        H = self.forward(q_, p_)
        dH = torch.autograd.grad(H.sum(), [q_, p_], create_graph=True)
        dq_dt = dH[1]   # dH/dp
        dp_dt = -dH[0]  # -dH/dq
        return dq_dt, dp_dt


class HNNPhaseSpaceDataset(Dataset):
    """Dataset of (q, p, dq/dt, dp/dt) from noisy orbit observations."""

    def __init__(self, data, v_stride=5, a_stride=10):
        dt_obs = data["t"][0][1] - data["t"][0][0]
        s_v = v_stride
        s_a = a_stride
        s = max(s_v, s_a)

        all_q, all_p, all_dq, all_dp = [], [], [], []

        for r_obs_np in data["r_obs"]:
            r_obs = np.array(r_obs_np)
            T = len(r_obs)
            if T <= 2 * s:
                continue

            # Velocity (central difference, stride s_v)
            v_est = (r_obs[2*s_v:] - r_obs[:-2*s_v]) / (2 * s_v * dt_obs)
            # Acceleration (second difference, stride s_a)
            a_est = (r_obs[2*s_a:] - 2 * r_obs[s_a:-s_a] + r_obs[:-2*s_a]) / (s_a * dt_obs) ** 2

            # Align: both need offset s from start
            # v_est is defined at indices [s_v, ..., T-1-s_v]
            # a_est is defined at indices [s_a, ..., T-1-s_a]
            # Use intersection: [s, ..., T-1-s]
            v_start = s - s_v
            v_end = T - 2 * s_v - (s - s_v)
            a_start = s - s_a
            a_end = T - 2 * s_a - (s - s_a)

            n_pts = min(len(v_est) - 2*(s - s_v), len(a_est) - 2*(s - s_a))
            if n_pts <= 0:
                continue

            # Positions at midpoints [s, ..., T-1-s]
            q = r_obs[s:s + n_pts]
            p = v_est[s - s_v:s - s_v + n_pts]   # p = m*v, m=1
            dq = p.copy()                          # dq/dt = v = p/m
            dp = a_est[s - s_a:s - s_a + n_pts]   # dp/dt = F = m*a

            all_q.append(q)
            all_p.append(p)
            all_dq.append(dq)
            all_dp.append(dp)

        self.q = torch.tensor(np.concatenate(all_q), dtype=torch.float32)
        self.p = torch.tensor(np.concatenate(all_p), dtype=torch.float32)
        self.dq = torch.tensor(np.concatenate(all_dq), dtype=torch.float32)
        self.dp = torch.tensor(np.concatenate(all_dp), dtype=torch.float32)

    def __len__(self):
        return len(self.q)

    def __getitem__(self, idx):
        return self.q[idx], self.p[idx], self.dq[idx], self.dp[idx]


def train_hnn(seed=42, n_epochs=200, hidden=128, lr=1e-3, batch_size=256, quiet=False,
              data_splits=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if data_splits is not None:
        train_data, val_data, test_data = data_splits
    else:
        rng = np.random.default_rng(seed)
        cfg = OrbitConfig()
        data = generate_dataset(cfg, sigma=1e-2, rng=rng)
        rng_split = np.random.default_rng(seed)
        train_data, val_data, test_data = train_val_test_split(data, rng=rng_split)

    train_ds = HNNPhaseSpaceDataset(train_data)
    val_ds = HNNPhaseSpaceDataset(val_data)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = HNN(hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not quiet:
        print(f"[HNN] Trainable parameters: {num_params}")
        print(f"[HNN] Training points: {len(train_ds)}, Val points: {len(val_ds)}")
        print(f"[HNN] Device: {device}")

    epoch_train_loss = []
    epoch_val_loss = []
    start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss_accum = 0.0
        n_batches = 0

        for q, p, dq_true, dp_true in train_loader:
            q, p = q.to(device), p.to(device)
            dq_true, dp_true = dq_true.to(device), dp_true.to(device)

            optimizer.zero_grad()
            dq_pred, dp_pred = model.time_derivative(q, p)
            loss = torch.mean((dq_pred - dq_true)**2 + (dp_pred - dp_true)**2)
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()
            n_batches += 1

        train_loss = train_loss_accum / n_batches

        model.eval()
        val_loss_accum = 0.0
        n_val = 0
        for q, p, dq_true, dp_true in val_loader:
            q, p = q.to(device), p.to(device)
            dq_true, dp_true = dq_true.to(device), dp_true.to(device)
            dq_pred, dp_pred = model.time_derivative(q, p)
            loss = torch.mean((dq_pred - dq_true)**2 + (dp_pred - dp_true)**2)
            val_loss_accum += loss.item()
            n_val += 1
        val_loss = val_loss_accum / max(n_val, 1)

        epoch_train_loss.append(train_loss)
        epoch_val_loss.append(val_loss)

        if not quiet and (epoch % 20 == 0 or epoch == 1):
            elapsed = time.time() - start_time
            print(f"[HNN] Epoch {epoch:03d} | train={train_loss:.4e} | val={val_loss:.4e} | time={elapsed:.1f}s")

    total_time = time.time() - start_time
    if not quiet:
        print(f"[HNN] Training finished in {total_time:.1f}s")

    # Evaluate: energy conservation on test orbits
    model.eval()
    dt_obs = test_data["t"][0][1] - test_data["t"][0][0]
    energy_vars = []

    for r_obs_np, t_np in zip(test_data["r_obs"], test_data["t"]):
        r_obs = np.array(r_obs_np)
        t = np.array(t_np)
        T = len(r_obs)
        s = 10
        if T <= 2 * s:
            continue

        q = torch.tensor(r_obs[s:-s], dtype=torch.float32).to(device)
        v_est = (r_obs[2*s:] - r_obs[:-2*s]) / (2 * s * dt_obs)
        n_pts = min(len(q), len(v_est))
        q = q[:n_pts]
        p = torch.tensor(v_est[:n_pts], dtype=torch.float32).to(device)

        with torch.no_grad():
            H = model(q, p).squeeze(-1).cpu().numpy()

        E_var = np.var(H) / (np.mean(H)**2 + 1e-12)
        energy_vars.append(E_var)

    mean_E_var = float(np.mean(energy_vars))

    # Rollout test: integrate using learned Hamiltonian
    rollout_errors = []
    for r_obs_np, t_np in zip(test_data["r_obs"][:3], test_data["t"][:3]):
        r_obs = np.array(r_obs_np)
        t = np.array(t_np)
        T = len(r_obs)
        s_v = 5

        q = torch.tensor(r_obs[0:1], dtype=torch.float32).to(device)
        v0 = (r_obs[s_v] - r_obs[0]) / (s_v * dt_obs)
        p = torch.tensor(v0[None, :], dtype=torch.float32).to(device)

        dt_int = dt_obs / 5.0
        n_steps = int((T - 1) * dt_obs / dt_int)
        trajectory = [q.detach().cpu().numpy()[0]]

        for step in range(n_steps):
            dq, dp = model.time_derivative(q, p)
            # Symplectic Euler (detach to prevent graph buildup)
            p = (p + dp * dt_int).detach()
            q = (q + dq * dt_int).detach()
            if (step + 1) % 5 == 0:
                trajectory.append(q.cpu().numpy()[0])

        trajectory = np.array(trajectory[:T])
        n_compare = min(len(trajectory), T)
        mse = np.mean((trajectory[:n_compare] - r_obs[:n_compare])**2)
        rollout_errors.append(float(mse))

    assumed_power_W = 200.0
    energy_kWh = assumed_power_W * total_time / 3.6e6

    results = {
        "num_params": num_params,
        "n_epochs": n_epochs,
        "total_time_s": total_time,
        "energy_kWh": energy_kWh,
        "final_train_loss": epoch_train_loss[-1],
        "final_val_loss": epoch_val_loss[-1],
        "mean_energy_variance": mean_E_var,
        "rollout_mse": rollout_errors,
        "epoch_train_loss": epoch_train_loss,
        "epoch_val_loss": epoch_val_loss,
        "interpretable": False,
        "energy_conserved_by_construction": True,
    }

    if not quiet:
        print(f"[HNN] Energy variance (test): {mean_E_var:.6f}")
        print(f"[HNN] Rollout MSE (3 orbits): {rollout_errors}")
        print(f"[HNN] Energy: {energy_kWh:.6f} kWh")

    return results


def main():
    parser = argparse.ArgumentParser(description="HNN baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=128)
    args = parser.parse_args()

    results = train_hnn(seed=args.seed, n_epochs=args.epochs, hidden=args.hidden)

    with open("hnn_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to hnn_results.json")


if __name__ == "__main__":
    main()
