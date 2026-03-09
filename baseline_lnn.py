"""
Lagrangian Neural Network (LNN) baseline for the Kepler benchmark.

Learns L(q, q_dot) as a neural network and enforces the Euler-Lagrange equations:
    d/dt(∂L/∂q_dot) = ∂L/∂q
which gives: (∂²L/∂q_dot²) * q_ddot = ∂L/∂q - (∂²L/∂q∂q_dot) * q_dot

Trained on noisy Kepler orbit data via acceleration prediction.

Key differences from MAL:
  - Black-box: no interpretable basis functions
  - Automatically respects Lagrangian mechanics structure
  - No symbolic force law discovery

Usage:
    python baseline_lnn.py
    python baseline_lnn.py --seed 42 --epochs 200
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


class LNN(nn.Module):
    """Lagrangian Neural Network: learns L(q, q_dot) and derives acceleration."""

    def __init__(self, input_dim=4, hidden=128, n_layers=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden), nn.Softplus()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Softplus()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, q, q_dot):
        x = torch.cat([q, q_dot], dim=-1)
        return self.net(x)

    def acceleration(self, q, q_dot):
        """Compute acceleration from Euler-Lagrange equation.

        q_ddot = (∂²L/∂q_dot²)^{-1} * (∂L/∂q - ∂²L/(∂q∂q_dot) * q_dot)

        For 2D: uses 2x2 Hessian inversion.
        """
        q = q.detach().requires_grad_(True)
        q_dot = q_dot.detach().requires_grad_(True)

        L = self.forward(q, q_dot)

        # First derivatives
        dL_dq, dL_dqdot = torch.autograd.grad(
            L.sum(), [q, q_dot], create_graph=True
        )

        # Hessian ∂²L/∂q_dot² (2x2 matrix per sample)
        batch_size = q.shape[0]
        dim = q.shape[1]  # 2

        # ∂²L/∂q_dot_i ∂q_dot_j
        hessian_qdot = torch.zeros(batch_size, dim, dim, device=q.device)
        for i in range(dim):
            grad_i = torch.autograd.grad(
                dL_dqdot[:, i].sum(), q_dot, create_graph=True
            )[0]
            hessian_qdot[:, i, :] = grad_i

        # Mixed Hessian ∂²L/∂q_i ∂q_dot_j
        mixed_hessian = torch.zeros(batch_size, dim, dim, device=q.device)
        for i in range(dim):
            grad_i = torch.autograd.grad(
                dL_dqdot[:, i].sum(), q, create_graph=True
            )[0]
            mixed_hessian[:, i, :] = grad_i

        # RHS: ∂L/∂q - (∂²L/∂q∂q_dot) * q_dot
        rhs = dL_dq - torch.bmm(mixed_hessian.transpose(1, 2), q_dot.unsqueeze(-1)).squeeze(-1)

        # Solve: hessian_qdot * q_ddot = rhs
        # Add small regularization for numerical stability
        reg = 1e-4 * torch.eye(dim, device=q.device).unsqueeze(0).expand(batch_size, -1, -1)
        hessian_reg = hessian_qdot + reg

        q_ddot = torch.linalg.solve(hessian_reg, rhs.unsqueeze(-1)).squeeze(-1)
        return q_ddot


class LNNDataset(Dataset):
    """Dataset of (q, q_dot, q_ddot) from noisy orbit observations."""

    def __init__(self, data, v_stride=5, a_stride=10):
        dt_obs = data["t"][0][1] - data["t"][0][0]
        s = max(v_stride, a_stride)

        all_q, all_qdot, all_qddot = [], [], []

        for r_obs_np in data["r_obs"]:
            r_obs = np.array(r_obs_np)
            T = len(r_obs)
            if T <= 2 * s:
                continue

            v_est = (r_obs[2*v_stride:] - r_obs[:-2*v_stride]) / (2 * v_stride * dt_obs)
            a_est = (r_obs[2*a_stride:] - 2 * r_obs[a_stride:-a_stride] + r_obs[:-2*a_stride]) / (a_stride * dt_obs) ** 2

            n_pts = min(len(v_est) - 2*(s - v_stride), len(a_est) - 2*(s - a_stride))
            if n_pts <= 0:
                continue

            q = r_obs[s:s + n_pts]
            qdot = v_est[s - v_stride:s - v_stride + n_pts]
            qddot = a_est[s - a_stride:s - a_stride + n_pts]

            all_q.append(q)
            all_qdot.append(qdot)
            all_qddot.append(qddot)

        self.q = torch.tensor(np.concatenate(all_q), dtype=torch.float32)
        self.qdot = torch.tensor(np.concatenate(all_qdot), dtype=torch.float32)
        self.qddot = torch.tensor(np.concatenate(all_qddot), dtype=torch.float32)

    def __len__(self):
        return len(self.q)

    def __getitem__(self, idx):
        return self.q[idx], self.qdot[idx], self.qddot[idx]


def train_lnn(seed=42, n_epochs=200, hidden=128, lr=1e-3, batch_size=256, quiet=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    rng = np.random.default_rng(seed)
    cfg = OrbitConfig()
    data = generate_dataset(cfg, sigma=1e-2, rng=rng)
    rng_split = np.random.default_rng(seed)
    train_data, val_data, test_data = train_val_test_split(data, rng=rng_split)

    train_ds = LNNDataset(train_data)
    val_ds = LNNDataset(val_data)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = LNN(hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not quiet:
        print(f"[LNN] Trainable parameters: {num_params}")
        print(f"[LNN] Training points: {len(train_ds)}, Val points: {len(val_ds)}")
        print(f"[LNN] Device: {device}")

    epoch_train_loss = []
    epoch_val_loss = []
    start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss_accum = 0.0
        n_batches = 0

        for q, qdot, qddot_true in train_loader:
            q, qdot, qddot_true = q.to(device), qdot.to(device), qddot_true.to(device)

            optimizer.zero_grad()
            try:
                qddot_pred = model.acceleration(q, qdot)
                loss = torch.mean((qddot_pred - qddot_true)**2)
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                train_loss_accum += loss.item()
            except Exception:
                # Singular Hessian — skip batch
                optimizer.step()
                train_loss_accum += 1e6
            n_batches += 1

        train_loss = train_loss_accum / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            val_loss_accum = 0.0
            n_val = 0
            for q, qdot, qddot_true in val_loader:
                q, qdot, qddot_true = q.to(device), qdot.to(device), qddot_true.to(device)
                try:
                    qddot_pred = model.acceleration(q, qdot)
                    loss = torch.mean((qddot_pred - qddot_true)**2)
                    val_loss_accum += loss.item()
                except Exception:
                    val_loss_accum += 1e6
                n_val += 1
            val_loss = val_loss_accum / max(n_val, 1)

        epoch_train_loss.append(train_loss)
        epoch_val_loss.append(val_loss)

        if not quiet and (epoch % 20 == 0 or epoch == 1):
            elapsed = time.time() - start_time
            print(f"[LNN] Epoch {epoch:03d} | train={train_loss:.4e} | val={val_loss:.4e} | time={elapsed:.1f}s")

    total_time = time.time() - start_time
    if not quiet:
        print(f"[LNN] Training finished in {total_time:.1f}s")

    # Evaluate energy conservation: compute Hamiltonian from Lagrangian
    # H = p·q_dot - L, where p = ∂L/∂q_dot
    model.eval()
    dt_obs = test_data["t"][0][1] - test_data["t"][0][0]
    energy_vars = []

    for r_obs_np in test_data["r_obs"]:
        r_obs = np.array(r_obs_np)
        T = len(r_obs)
        s = 10
        if T <= 2 * s:
            continue

        q = torch.tensor(r_obs[s:-s], dtype=torch.float32).to(device)
        v_est = (r_obs[2*s:] - r_obs[:-2*s]) / (2 * s * dt_obs)
        n_pts = min(len(q), len(v_est))
        q = q[:n_pts].requires_grad_(True)
        qdot = torch.tensor(v_est[:n_pts], dtype=torch.float32).to(device).requires_grad_(True)

        L = model(q, qdot)
        dL_dqdot = torch.autograd.grad(L.sum(), qdot, create_graph=False)[0]

        # Hamiltonian H = p · q_dot - L
        H = torch.sum(dL_dqdot * qdot, dim=-1) - L.squeeze(-1)
        H = H.detach().cpu().numpy()

        E_var = np.var(H) / (np.mean(H)**2 + 1e-12)
        energy_vars.append(E_var)

    mean_E_var = float(np.mean(energy_vars))

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
        "epoch_train_loss": epoch_train_loss,
        "epoch_val_loss": epoch_val_loss,
        "interpretable": False,
        "energy_conserved_by_construction": True,
    }

    if not quiet:
        print(f"[LNN] Energy variance (test): {mean_E_var:.6f}")
        print(f"[LNN] Energy: {energy_kWh:.6f} kWh")

    return results


def main():
    parser = argparse.ArgumentParser(description="LNN baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=128)
    args = parser.parse_args()

    results = train_lnn(seed=args.seed, n_epochs=args.epochs, hidden=args.hidden)

    with open("lnn_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to lnn_results.json")


if __name__ == "__main__":
    main()
