"""RGSS: co-estimation of annotation reliability and target construction (Sec. 3).

For the two co-trained models M_1, M_2 and every training sample i:
    l_i^(k)  = Dice(sigma(M_k(x_i)), y_i)                                   per-sample loss
    r_i^(k)  = p(clean | l_i^(k'))  with a 2-component GMM on the losses of the peer k' != k   (Eq. 2)
    J_clean  = {i : r_i^(k) >= tau}
    y^_i     = y_i                         if i in J_clean
             = R_phi(x_i, y_i, T_i),  T_i = T_min + (1 - r_i)(T_max - T_min)    otherwise      (Eq. 3)
    y~_i     = beta_i y^_i + (1 - beta_i) sigma(M_k'(x_i)),
    beta_i   = r_i if i in J_clean else alpha(e) = (e_max - e) / (e_max - e_0)                (Eq. 4)
"""

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader

from rgss.segmentation import dice_loss

# GMM fitting settings (as in DivideMix)
GMM_KW = dict(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)


@torch.no_grad()
def per_sample_losses(model, dataset, batch_size, num_workers, device="cuda"):
    """l_i = Dice(sigma(M(x_i)), y_i) on the given annotations, in dataset order."""
    model.eval()
    out = []
    for b in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers):
        prob = model(b["image"].to(device)).sigmoid()
        out.append(dice_loss(prob, b["mask"].to(device), reduction="none").cpu())
    return torch.cat(out).numpy()


def clean_probability(losses, seed):
    """Eq. (2): posterior of the low-loss component of a 2-component GMM fitted to the losses."""
    l = (losses - losses.min()) / (losses.max() - losses.min() + 1e-8)
    gmm = GaussianMixture(random_state=seed, **GMM_KW).fit(l.reshape(-1, 1))
    return gmm.predict_proba(l.reshape(-1, 1))[:, int(np.argmin(gmm.means_.ravel()))]


def refinement_steps(r, t_min, t_max):
    """T_i = T_min + (1 - r_i)(T_max - T_min), rounded to an integer number of reverse steps."""
    return np.rint(t_min + (1.0 - r) * (t_max - t_min)).astype(np.int64)


@torch.no_grad()
def construct_targets(dataset, reliability, peer, refiner, tau, t_min, t_max, alpha, batch_size, num_workers, device="cuda"):
    """Soft targets y~ (N, 1, S, S) for one model, given its reliabilities r (N,) and the peer model."""
    peer.eval()
    clean = reliability >= tau
    targets = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    for b in loader:
        idx = b["index"].numpy()
        img, y = b["image"].to(device), b["mask"].to(device)
        r = torch.as_tensor(reliability[idx], dtype=torch.float32, device=device).view(-1, 1, 1, 1)
        is_clean = torch.as_tensor(clean[idx], device=device).view(-1, 1, 1, 1)
        y_hat = y.clone()
        noisy = np.where(~clean[idx])[0]
        if len(noisy):  # Eq. (3)
            steps = refinement_steps(reliability[idx][noisy], t_min, t_max)
            refined = refiner.refine(img[noisy], y[noisy], torch.as_tensor(steps, device=device))
            y_hat[noisy] = (refined > 0.5).float()
        beta = torch.where(is_clean, r, torch.full_like(r, alpha))  # Eq. (4)
        targets.append((beta * y_hat + (1 - beta) * peer(img).sigmoid()).cpu())
    return torch.cat(targets)
