
import torch
import torch.nn as nn

G = 1.0
M = 1.0

class NoetherForceBasis(nn.Module):
    """
    Central-force Noether network enforcing SO(2) rotational symmetry.
    Force is always radial: f(r_vec) = f_r(r) * (-r_hat).
    """
    # Default 5-term radial basis library
    DEFAULT_BASIS_FNS = None  # built in get_phi when basis_fns is None

    def __init__(self, n_terms=5, basis_fns=None):
        super().__init__()
        if basis_fns is not None:
            self.basis_fns = basis_fns
            self.n_terms = len(basis_fns)
        else:
            self.basis_fns = None
            self.n_terms = n_terms
        self.theta = nn.Parameter(torch.randn(self.n_terms) * 0.1)
        self.A_logits = nn.Parameter(torch.ones(self.n_terms))
        self.tau = 1.0  # gate temperature, set externally for annealing

    def get_phi(self, r):
        eps = 1e-8
        if self.basis_fns is not None:
            phi = [fn(r.clamp(min=eps)) for fn in self.basis_fns]
            return torch.cat(phi, dim=-1)
        phi = []
        phi.append(1.0 / (r**2 + eps))   # 1/r^2
        phi.append(1.0 / (r + eps))      # 1/r
        phi.append(r)                    # r
        phi.append(torch.ones_like(r))   # const
        phi.append(1.0 / (r**3 + eps))   # 1/r^3
        return torch.cat(phi[:self.n_terms], dim=-1)

    def forward(self, r_vec):
        r_mag = torch.linalg.norm(r_vec, dim=-1, keepdim=True)
        phi = self.get_phi(r_mag)
        A = torch.softmax(self.A_logits / self.tau, dim=0)
        f_radial_mag = torch.sum(A * self.theta * phi, dim=-1, keepdim=True)
        r_hat = r_vec / (r_mag + 1e-12)
        f_vec = -f_radial_mag * r_hat
        return f_vec, A

class MinActionNet(nn.Module):
    def __init__(self, dt, n_terms=5, basis_fns=None):
        super().__init__()
        self.dt = dt
        self.force_basis = NoetherForceBasis(n_terms=n_terms, basis_fns=basis_fns)

    def integrate_step(self, r, v):
        f, _ = self.force_basis(r)
        v_half = v + 0.5 * self.dt * f
        r_new = r + self.dt * v_half
        f_new, _ = self.force_basis(r_new)
        v_new = v_half + 0.5 * self.dt * f_new
        return r_new, v_new

    def rollout(self, r0, v0, n_steps):
        rs = [r0]
        vs = [v0]
        r, v = r0, v0
        for _ in range(n_steps):
            r, v = self.integrate_step(r, v)
            rs.append(r)
            vs.append(v)
        return torch.stack(rs, dim=1), torch.stack(vs, dim=1)

def orbital_energy(r, v, potential='gravity', k=1.0):
    r_norm = torch.linalg.norm(r, dim=-1)
    v_norm2 = torch.sum(v * v, dim=-1)
    KE = 0.5 * v_norm2
    if potential == 'gravity':
        PE = -G * M / (r_norm + 1e-6)
    elif potential == 'hooke':
        PE = 0.5 * k * r_norm**2
    else:
        raise ValueError(f"Unknown potential: {potential}")
    return KE + PE

def minaction_loss(
    model,
    r_obs,
    dt_obs,
    alpha_I=1.0,
    alpha_E=0.1,
    lambda_accel=1.0,
    lambda_comp=0.01,
    lambda_arch=0.5,
    lambda_sym=1.0,
    stride=10,
    potential='gravity',
    k=1.0,
):
    device = r_obs.device
    B, T, _ = r_obs.shape
    n_sub = int(round(dt_obs / model.dt))
    assert n_sub >= 1

    # --- Teacher-forced integration loss ---
    # Identifies which basis function by evaluating force at observed positions.
    r = r_obs[:, 0]
    # Use wider baseline for initial velocity to reduce noise
    s_v = min(5, T - 1)
    v = (r_obs[:, s_v] - r_obs[:, 0]) / (s_v * dt_obs)

    L_traj = 0.0
    collected_r = []
    collected_v = []

    for t_idx in range(T - 1):
        for _ in range(n_sub):
            r, v = model.integrate_step(r, v)
        L_traj += torch.mean((r - r_obs[:, t_idx + 1]) ** 2)
        collected_r.append(r)
        collected_v.append(v)
        r = r_obs[:, t_idx + 1]

    L_traj = L_traj / (T - 1)

    # --- Direct acceleration matching loss ---
    # Use wider stencil to reduce noise: a ≈ (r[i+s] - 2r[i] + r[i-s]) / (s*dt)²
    # Noise variance drops by s^4 while signal is preserved for slowly-varying forces.
    s = stride  # noise reduction factor = s^4
    a_est = (r_obs[:, 2*s:] - 2 * r_obs[:, s:-s] + r_obs[:, :-2*s]) / (s * dt_obs) ** 2
    r_mid = r_obs[:, s:-s]
    n_accel = T - 2 * s
    r_flat = r_mid.reshape(-1, 2)
    f_pred, _ = model.force_basis(r_flat)
    f_pred = f_pred.reshape(B, n_accel, 2)
    L_accel = torch.mean((f_pred - a_est) ** 2)

    L_I = L_traj + lambda_accel * L_accel

    # Energy conservation from model-integrated velocities
    r_stack = torch.stack(collected_r, dim=0).reshape(-1, 2)
    v_stack = torch.stack(collected_v, dim=0).reshape(-1, 2)
    E = orbital_energy(r_stack, v_stack, potential=potential, k=k)
    E_mean = torch.mean(E)
    L_sym = torch.mean((E - E_mean) ** 2)

    # Sparsity losses
    A_logits = model.force_basis.A_logits
    tau = model.force_basis.tau
    A = torch.softmax(A_logits / tau, dim=0)

    theta = model.force_basis.theta
    L_comp = torch.mean(torch.abs(A * theta))

    entropy = -torch.sum(A * torch.log(A + 1e-8))
    L_arch = entropy

    L_E = lambda_sym * L_sym + lambda_comp * L_comp + lambda_arch * L_arch
    loss = alpha_I * L_I + alpha_E * L_E
    loss_dict = dict(
        total=loss.detach().item(),
        L_I=L_I.detach().item(),
        L_traj=L_traj.detach().item(),
        L_accel=L_accel.detach().item(),
        L_E=L_E.detach().item(),
        L_sym=L_sym.detach().item(),
        L_comp=L_comp.detach().item(),
        L_arch=L_arch.detach().item(),
    )
    return loss, loss_dict


def calibrate_theta(model, train_data, dt_obs):
    """Post-training: set theta magnitude by least-squares on acceleration data.

    Training with sparsity + noise identifies the correct gate but biases theta
    toward zero.  This step fixes the magnitude using the same wide-stencil
    acceleration estimates, projected onto the radial direction.
    """
    A = torch.softmax(model.force_basis.A_logits / model.force_basis.tau, dim=0)
    dominant = torch.argmax(A).item()

    s = 10  # same stencil as training
    all_a_radial = []
    all_phi_dom = []

    for r_obs_np in train_data["r_obs"]:
        r_obs = torch.tensor(r_obs_np, dtype=torch.float32)
        T = r_obs.shape[0]
        if T <= 2 * s:
            continue

        a_est = (r_obs[2*s:] - 2 * r_obs[s:-s] + r_obs[:-2*s]) / (s * dt_obs) ** 2
        r_mid = r_obs[s:-s]
        r_mag = torch.linalg.norm(r_mid, dim=-1, keepdim=True)
        r_hat = r_mid / (r_mag + 1e-12)

        # Model force: F = -theta * phi(r) * r_hat  =>  a · (-r_hat) = theta * phi(r)
        a_radial = -torch.sum(a_est * r_hat, dim=-1)  # positive for attractive force

        phi = model.force_basis.get_phi(r_mag)  # (N, n_terms)
        phi_dom = phi[:, dominant]               # (N,)

        all_a_radial.append(a_radial)
        all_phi_dom.append(phi_dom)

    all_a_radial = torch.cat(all_a_radial)
    all_phi_dom = torch.cat(all_phi_dom)

    # theta_opt = argmin sum((theta * phi - a_radial)^2) = sum(phi * a_radial) / sum(phi^2)
    theta_opt = torch.sum(all_phi_dom * all_a_radial) / torch.sum(all_phi_dom ** 2)

    with torch.no_grad():
        model.force_basis.theta[dominant] = theta_opt
    print(f"  Calibrated theta[{dominant}] = {theta_opt.item():.4f}")
