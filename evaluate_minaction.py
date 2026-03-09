
import torch
import numpy as np
import matplotlib.pyplot as plt

from minaction_model import MinActionNet

def plot_discovery_results(model, test_orbit_data, energy_kwh):
    model.eval()
    r_clean = torch.tensor(test_orbit_data['r_clean'], dtype=torch.float32)
    v_clean = torch.tensor(test_orbit_data['v_clean'], dtype=torch.float32)
    t = test_orbit_data['t']

    dev = next(model.parameters()).device
    r0 = r_clean[0:1].to(dev)
    v0 = v_clean[0:1].to(dev)

    dt_obs = t[1] - t[0]
    n_sub = int(round(dt_obs / model.dt))
    n_rollout = (len(t) - 1) * n_sub

    with torch.no_grad():
        r_pred, v_pred = model.rollout(r0, v0, n_steps=n_rollout)
        # Subsample to observation cadence for plotting
        r_pred_t = r_pred.squeeze(0)[::n_sub].cpu()
        r_pred = [[r_pred_t[i, 0].item(), r_pred_t[i, 1].item()] for i in range(r_pred_t.shape[0])]
        gates = torch.softmax(model.force_basis.A_logits / model.force_basis.tau, dim=0).cpu().tolist()
        weights = model.force_basis.theta.cpu().tolist()

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    r_pred_x = [p[0] for p in r_pred]
    r_pred_y = [p[1] for p in r_pred]
    r_clean_x = r_clean[:, 0].tolist()
    r_clean_y = r_clean[:, 1].tolist()
    axs[0].plot(r_clean_x, r_clean_y, 'k--', label='Ground Truth', alpha=0.5)
    axs[0].plot(r_pred_x, r_pred_y, 'r-', label='MinAction.Net Prediction')
    axs[0].set_title(f"Orbit Rollout\n(Energy Cost: {energy_kwh:.6f} kWh)")
    axs[0].legend()
    axs[0].axis('equal')

    labels = ['1/r^2', '1/r', 'r', '1', '1/r^3']
    axs[1].bar(labels, gates, color='teal')
    axs[1].set_ylim(0, 1.1)
    axs[1].set_title("Architecture Gates A(t)\n(Parsimony/Discreteness)")
    axs[1].set_ylabel("Selection Probability")

    axs[2].bar(labels, weights, color='orange')
    axs[2].set_title("Learned Coefficients θ\n(Force Magnitude)")
    axs[2].set_ylabel("Value")

    plt.tight_layout()
    plt.savefig("discovery_report.png")
    print("Report saved as discovery_report.png")
    plt.show()
