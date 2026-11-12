"""Residual refinement: pixel-bank consistency during reverse diffusion.

Generation is residual-based: a PreNet predicts a coarse image ``IP`` and the
denoising diffusion network samples a residual ``R``; the output is ``IP + R``.
ResTTA refines the *residual* at test time by inserting a few gradient steps
towards neighbourhood consistency (a pixel-bank graph prior) between reverse
diffusion steps.  No ground truth and no re-training are used.

Two entry points are provided:

``tta_p_sample_loop``
    Full reverse-diffusion loop with intra-step graph refinement of ``R``.
    Multi-scale banks are mixed with a schedule (small windows early, large
    windows late).  The Proximal term keeps the refined residual close to the
    DDPM posterior sample; the frequency-domain filter protects low-frequency
    structure while allowing high-frequency correction.

``refine_prenet_ip``
    A deterministic K-step graph refinement applied directly to ``IP`` (used by
    the PreNet-only ablation).  It has no reverse-diffusion projection, so it is
    simply an output-side graph regulariser.
"""

import numpy as np
import torch
from tqdm import tqdm

from pixel_bank import apply_bilateral_slice, build_pixel_bank_slice_2d


def neighborhood_consistency_loss(x, pb_indices_t):
    """L_nc(x) = mean_p || x[p] - mean_{q in N(p)} x[q] ||^2.

    ``pb_indices_t`` is ``(num_rows, H, W)`` with flattened neighbour indices.
    """
    H, W = x.shape[2], x.shape[3]
    num_rows = pb_indices_t.shape[0]
    x_flat = x.reshape(-1)
    idx_flat = pb_indices_t.reshape(num_rows, -1)
    neighbor_vals = x_flat[idx_flat]
    neighbor_mean = neighbor_vals.mean(dim=0)
    return ((x_flat - neighbor_mean) ** 2).mean()


def frequency_filtered_gradient(grad, low_radius=16, low_scale=0.3):
    """Attenuate low-frequency gradient components in the Fourier domain.

    ``low_scale=1.0`` disables the filter; ``low_radius`` sets the transition
    radius between attenuated and preserved frequencies.
    """
    if low_scale >= 1.0:
        return grad
    H, W = grad.shape[2], grad.shape[3]
    G = torch.fft.fft2(grad)
    G_centered = torch.fft.fftshift(G)
    cy, cx = H // 2, W // 2
    y = torch.arange(H, device=grad.device).float()
    x = torch.arange(W, device=grad.device).float()
    dist = torch.sqrt((y.unsqueeze(1) - cy) ** 2 +
                      (x.unsqueeze(0) - cx) ** 2)
    transition = max(2.0, low_radius / 4.0)
    weight = low_scale + (1.0 - low_scale) * torch.sigmoid(
        (dist - low_radius) / transition)
    weight = weight.unsqueeze(0).unsqueeze(0)
    G_filtered = torch.fft.ifftshift(G_centered * weight)
    return torch.fft.ifft2(G_filtered).real


def _select_pb(pb_input, pb_scale, device):
    """Normalise a single-scale tensor / multi-scale dict to ``{key: tensor}``."""
    if isinstance(pb_input, dict):
        pb_in = {k: v.to(device) if v.device != device else v
                 for k, v in pb_input.items()}
        if pb_scale == "all":
            return pb_in
        if pb_scale == "medium":
            key = next((k for k in pb_in if "17" in k), None)
        else:
            key = next((k for k in pb_in if str(pb_scale) in k), None)
        if key is None:
            key = next((k for k in pb_in if "17" in k), list(pb_in.keys())[0])
        return {key: pb_in[key]}

    pb_t = pb_input.to(device) if pb_input.device != device else pb_input
    return {"s17": pb_t}


def refine_prenet_ip(IP, pb_input, K, lr,
                     freq_low_radius=16, freq_low_scale=0.3,
                     pb_scale="medium"):
    """Apply K proximal graph-update steps to the PreNet prediction ``IP``.

    Objective per step::

        L = L_nc(x; PB)
        x <- x - lr * freq_filter(grad_x L)

    Args:
        IP: ``(1, 1, H, W)`` coarse prediction tensor.
        pb_input: single-scale tensor or multi-scale dict of banks.
        K, lr: number of steps and step size (no-op when either is non-positive).
        pb_scale: ``'medium'`` (17), ``'all'`` or a window size such as 9/33.
    """
    if K <= 0 or lr <= 0:
        return IP

    device = IP.device
    pb_dict = _select_pb(pb_input, pb_scale, device)

    x = IP.detach().clone()
    for _ in range(K):
        with torch.enable_grad():
            xg = x.detach().requires_grad_(True)
            if len(pb_dict) == 1:
                loss_nc = neighborhood_consistency_loss(
                    xg, next(iter(pb_dict.values())))
            else:
                loss_nc = sum(neighborhood_consistency_loss(xg, pb)
                              for pb in pb_dict.values()) / len(pb_dict)
            grad = torch.autograd.grad(loss_nc, xg)[0]

        grad = frequency_filtered_gradient(
            grad, low_radius=freq_low_radius, low_scale=freq_low_scale)
        x = x.detach() - lr * grad
    return x


@torch.no_grad()
def tta_p_sample_loop(netG, condition_x, ax_feature, fr_feature, pb_input,
                      constraint_start_ratio=1.0,
                      lr_mid=1e-3, lr_late=1e-4,
                      K_mid=2, K_late=5, lam=0.5,
                      IP=None, sigma_est=0.0,
                      num_neighbors=16, window_size=17,
                      freq_low_radius=16, freq_low_scale=0.3):
    """Reverse diffusion for the residual with pixel-bank consistency TTA.

    Args:
        netG: ``GaussianDiffusion`` (DenoiseNet) instance.
        condition_x: low-dose conditioning slice ``(B, 1, H, W)``.
        ax_feature, fr_feature: guide features from the six guide networks.
        pb_input: single-scale tensor or multi-scale dict.
        constraint_start_ratio: fraction of the trajectory (from the noisy end)
            during which graph refinement is active.
        lr_mid / lr_late: step sizes in the main and late stages.
        K_mid / K_late: graph steps per diffusion step in each stage.
        lam: weight of the proximal term relative to L_nc.
        IP: PreNet prediction, used to build the hybrid bank.
        sigma_est: bilateral sigma for the hybrid-bank estimate (0 disables).

    Returns:
        The refined residual tensor with the same shape as ``condition_x``.
    """
    device = netG.betas.device
    num_timesteps = netG.num_timesteps
    condition_x = condition_x.to(device)

    if isinstance(pb_input, dict):
        pb_ms = {k: v.to(device) for k, v in pb_input.items()}
        use_multi_scale = True
        pb_medium_key = next((k for k in pb_ms if "17" in k),
                             list(pb_ms.keys())[0])
    else:
        pb_ms = {"s17": pb_input.to(device)}
        use_multi_scale = False
        pb_medium_key = "s17"

    constraint_start_step = int(num_timesteps * constraint_start_ratio)
    mid_end_step = int(constraint_start_step * 0.4)

    img = torch.randn_like(condition_x).to(device)
    pb_active = dict(pb_ms)
    hybrid_built = False

    for t_val in tqdm(reversed(range(0, num_timesteps)),
                      desc="ResTTA sampling", total=num_timesteps,
                      leave=False):
        batch_size = img.shape[0]
        noise_level = torch.FloatTensor(
            [netG.sqrt_alphas_cumprod_prev[t_val + 1]]
        ).repeat(batch_size, 1).to(device)

        denoise_input = torch.cat([condition_x, img], dim=1)
        noise_pred = netG.denoise_fn(denoise_input, noise_level,
                                     ax_feature, fr_feature)

        x_recon = (netG.sqrt_recip_alphas_cumprod[t_val] * img -
                   netG.sqrt_recipm1_alphas_cumprod[t_val] * noise_pred)

        posterior_mean = (netG.posterior_mean_coef1[t_val] * x_recon +
                          netG.posterior_mean_coef2[t_val] * img)
        posterior_log_variance = netG.posterior_log_variance_clipped[t_val]

        noise = torch.randn_like(img) if t_val > 0 else torch.zeros_like(img)
        img = posterior_mean + noise * (0.5 * posterior_log_variance).exp()

        steps_from_end = num_timesteps - 1 - t_val

        if steps_from_end < constraint_start_step:
            # Hybrid bank: half LPET-derived neighbours + half neighbours from
            # the current estimate (IP + img), rebuilt once per slice.
            if (not hybrid_built and sigma_est is not None and sigma_est > 0
                    and IP is not None):
                pb_med = pb_ms[pb_medium_key]
                half_rows = pb_med.shape[0] // 2

                i_est = (IP + img).squeeze().cpu().numpy().astype(np.float64)
                i_est_blur = apply_bilateral_slice(i_est, sigma_spatial=sigma_est)

                pb_est_np = build_pixel_bank_slice_2d(
                    i_est_blur, window_size=window_size,
                    num_neighbors=num_neighbors, num_rows=half_rows,
                    device=device)
                pb_est_t = torch.from_numpy(pb_est_np).long().to(device)
                pb_active[pb_medium_key] = torch.cat(
                    [pb_med[:half_rows], pb_est_t], dim=0)
                hybrid_built = True

            img_ddpm = img.clone()

            if steps_from_end < mid_end_step:
                K, lr = K_late, lr_late
            else:
                K, lr = K_mid, lr_mid

            progress = 1.0 - steps_from_end / max(constraint_start_step, 1)
            w_small = 0.3 + 0.7 * progress
            w_medium = 0.6 + 0.4 * (1.0 - progress)
            w_large = 1.0 - progress
            w_sum = w_small + w_medium + w_large

            for _ in range(K):
                with torch.enable_grad():
                    img_grad = img.detach().requires_grad_(True)

                    if use_multi_scale:
                        loss_nc = 0.0
                        for key, pb_t in pb_active.items():
                            ws_val = int(key.replace("s", ""))
                            if ws_val <= 9:
                                w = w_small
                            elif ws_val >= 33:
                                w = w_large
                            else:
                                w = w_medium
                            loss_nc += w * neighborhood_consistency_loss(
                                img_grad, pb_t)
                        loss_nc = loss_nc / w_sum
                    else:
                        loss_nc = neighborhood_consistency_loss(
                            img_grad, pb_active[pb_medium_key])

                    loss_prox = ((img_grad - img_ddpm) ** 2).mean()

                    with torch.no_grad():
                        r = loss_prox.item() / (loss_nc.item() + 1e-8)
                        lam_dynamic = max(0.0, min(2.0, lam * r))

                    loss = loss_nc + lam_dynamic * loss_prox
                    grad = torch.autograd.grad(loss, img_grad)[0]

                grad = frequency_filtered_gradient(
                    grad, low_radius=freq_low_radius, low_scale=freq_low_scale)
                img = img.detach() - lr * grad

    return img
