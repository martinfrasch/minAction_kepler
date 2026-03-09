
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data_kepler import OrbitConfig, generate_dataset, train_val_test_split
from minaction_model import MinActionNet, minaction_loss, calibrate_theta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def estimate_period_from_traj(r_clean, t):
    x = r_clean[:, 0] - np.mean(r_clean[:, 0])
    N = len(x)
    corr = np.correlate(x, x, mode="full")
    corr = corr[N - 1:]  # positive lags starting from lag=0
    # Normalize by overlapping sample count to remove length bias
    corr = corr / np.arange(N, 0, -1)
    corr = corr / corr[0]
    # Walk past the initial decay from lag=0 to find the first trough
    i = 1
    while i < N - 1 and corr[i] <= corr[i - 1]:
        i += 1
    # Then walk to the next peak (first periodic maximum)
    while i < N - 1 and corr[i] <= corr[i + 1]:
        i += 1
    T_est = t[i] - t[0]
    return T_est

def fit_kepler_exponent(a_vals, T_vals):
    y = np.log(T_vals ** 2)
    x = np.log(a_vals)
    A = np.vstack([x, np.ones_like(x)]).T
    p_hat, logC_hat = np.linalg.lstsq(A, y, rcond=None)[0]
    C_hat = np.exp(logC_hat)
    return p_hat, C_hat

def main():
    cfg = OrbitConfig()
    sigma = 1e-2
    data = generate_dataset(cfg, sigma=sigma)
    train_data, val_data, test_data = train_val_test_split(data)

    max_T = 200
    train_ds = OrbitDataset(train_data, max_T=max_T)
    val_ds = OrbitDataset(val_data, max_T=max_T)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)

    dt_obs_example = train_data["t"][0][1] - train_data["t"][0][0]
    model_dt = dt_obs_example / 5.0
    model = MinActionNet(dt=model_dt).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = 200
    print_every = 10

    warmup_epochs = 50
    start_time = time.time()
    for epoch in range(1, n_epochs + 1):
        # Two-phase schedule: warmup then sparsify
        if epoch <= warmup_epochs:
            alpha_E = 0.01
            tau = 1.0
        else:
            progress = (epoch - warmup_epochs) / (n_epochs - warmup_epochs)
            alpha_E = 0.01 + 0.99 * progress
            tau = 1.0 * (0.05 / 1.0) ** progress  # exponential decay 1.0 → 0.05
        model.force_basis.tau = tau

        model.train()
        train_loss_accum = 0.0
        last_loss_dict = None
        for r_obs, dt_obs in train_loader:
            r_obs = r_obs.to(device)
            dt_obs = dt_obs.to(device)
            optimizer.zero_grad()
            loss, loss_dict = minaction_loss(model, r_obs, dt_obs.item(), alpha_E=alpha_E)
            loss.backward()
            optimizer.step()
            train_loss_accum += loss.item() * r_obs.size(0)
            last_loss_dict = loss_dict

        train_loss = train_loss_accum / len(train_ds)

        model.eval()
        with torch.no_grad():
            val_loss_accum = 0.0
            for r_obs, dt_obs in val_loader:
                r_obs = r_obs.to(device)
                dt_obs = dt_obs.to(device)
                loss, _ = minaction_loss(model, r_obs, dt_obs.item(), alpha_E=alpha_E)
                val_loss_accum += loss.item() * r_obs.size(0)
            val_loss = val_loss_accum / len(val_ds)

        if epoch % print_every == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:03d} | train={train_loss:.4e} | val={val_loss:.4e} | alpha_E={alpha_E:.3f} | tau={tau:.3f} | time={elapsed:.1f}s")
            if last_loss_dict:
                print(f"  L_traj={last_loss_dict['L_traj']:.4e}  L_accel={last_loss_dict['L_accel']:.4e}  L_comp={last_loss_dict['L_comp']:.4e}  L_arch={last_loss_dict['L_arch']:.4e}")
            with torch.no_grad():
                gates = torch.softmax(model.force_basis.A_logits / model.force_basis.tau, dim=0)
                print("  Gates A:", [f"{g:.4f}" for g in gates.cpu().tolist()])
                print("  Theta:  ", [f"{t:.4f}" for t in model.force_basis.theta.cpu().tolist()])

    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.1f}s")

    # Calibrate theta magnitude via least-squares on wide-stencil acceleration estimates.
    # Training identifies the correct gate but L1 penalty biases theta toward zero.
    dt_obs_cal = train_data["t"][0][1] - train_data["t"][0][0]
    calibrate_theta(model, train_data, dt_obs_cal)
    with torch.no_grad():
        gates = torch.softmax(model.force_basis.A_logits / model.force_basis.tau, dim=0)
        print("  Final theta:", [f"{t:.4f}" for t in model.force_basis.theta.cpu().tolist()])
        print("  Final gates:", [f"{g:.4f}" for g in gates.cpu().tolist()])
    assumed_power_W = 200.0
    energy_J = assumed_power_W * total_time
    energy_kWh = energy_J / 3.6e6
    print(f"Approx. training energy: {energy_kWh:.4f} kWh (assuming {assumed_power_W}W)")

    a_vals = test_data["a"]
    T_clean = []
    for r_clean, t in zip(test_data["r_clean"], test_data["t"]):
        T_est = estimate_period_from_traj(np.array(r_clean), np.array(t))
        T_clean.append(T_est)
    T_clean = np.array(T_clean)

    p_hat, C_hat = fit_kepler_exponent(a_vals, T_clean)
    print(f"Fitted Kepler exponent p_hat (clean orbits) = {p_hat:.4f}")
    print(f"Fitted constant C_hat (clean orbits) = {C_hat:.4f}")

    return model, energy_kWh, test_data

if __name__ == "__main__":
    main()
