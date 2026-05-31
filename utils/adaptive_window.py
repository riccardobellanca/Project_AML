

import math
import torch
import torch.nn.functional as F
from typing import Tuple


# ---------------------------------------------------------------------------
# Entropy of a distribution
# ---------------------------------------------------------------------------

def distribution_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:

    log_p   = torch.log(probs.clamp(min=eps))
    entropy = -(probs * log_p).sum(dim=-1)
    # Normalise by log(N) so that output ∈ [0, 1]
    max_entropy = math.log(probs.shape[-1])
    return entropy / (max_entropy + eps)


# ---------------------------------------------------------------------------
# Window extraction around a peak
# ---------------------------------------------------------------------------

def extract_window(
    sim_map: torch.Tensor,   # (h, w)
    peak_y:  int,
    peak_x:  int,
    radius:  int,
) -> Tuple[torch.Tensor, int, int, int, int]:
    """
    Extract a (2r+1) × (2r+1) window centred at (peak_y, peak_x),
    clamping to image boundaries.

    Returns:
        window:   extracted patch (variable size at boundaries).
        y0, y1:   row slice [y0, y1) in the original map.
        x0, x1:   col slice [x0, x1) in the original map.
    """
    h, w = sim_map.shape
    y0 = max(0,     peak_y - radius)
    y1 = min(h,     peak_y + radius + 1)
    x0 = max(0,     peak_x - radius)
    x1 = min(w,     peak_x + radius + 1)
    return sim_map[y0:y1, x0:x1], y0, y1, x0, x1


# ---------------------------------------------------------------------------
# Adaptive Window Soft-Argmax  (main API)
# ---------------------------------------------------------------------------

def adaptive_window_softargmax(
    sim_row: torch.Tensor,       # (h*w,) raw similarity scores for one keypoint
    h: int,
    w: int,
    temperature: float  = 0.02,  # softmax temperature inside the window
    min_radius:  int    = 2,     # minimum window half-size
    max_radius:  int    = 7,     # maximum window half-size
    entropy_threshold_low:  float = 0.2,   # below → use min_radius
    entropy_threshold_high: float = 0.7,   # above → use max_radius
) -> Tuple[torch.Tensor, float]:

    # --- 1. Full-map soft distribution ---
    probs_flat  = F.softmax(sim_row.float(), dim=0)          # (h*w,)
    ent         = distribution_entropy(probs_flat).item()

    # Salvaguardia contro instabilità numeriche dovute ad AMP (Float16)
    if math.isnan(ent):
        ent = 1.0

    # --- 2. Adaptive radius ---
    if math.isnan(ent) or math.isinf(ent):
        radius = max_radius
    elif ent <= entropy_threshold_low:
        radius = min_radius
    elif ent >= entropy_threshold_high:
        radius = max_radius
    else:
        # Linear interpolation between thresholds
        t = (ent - entropy_threshold_low) / (entropy_threshold_high - entropy_threshold_low)
        if math.isnan(t) or math.isinf(t):
            radius = max_radius
        else:
            radius = int(round(min_radius + t * (max_radius - min_radius)))

    # --- 3. Coarse argmax peak ---
    peak_flat = probs_flat.argmax().item()
    peak_y    = peak_flat // w
    peak_x    = peak_flat % w

    # --- 4. Window soft-argmax ---
    sim_map = sim_row.reshape(h, w)
    window, y0, _, x0, _ = extract_window(sim_map, peak_y, peak_x, radius)

    win_probs = F.softmax(window.reshape(-1) / temperature, dim=0)
    wh, ww    = window.shape

    # Build local coordinate grid
    ys = torch.arange(wh, device=sim_row.device, dtype=torch.float32) + y0
    xs = torch.arange(ww, device=sim_row.device, dtype=torch.float32) + x0
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    gx = gx.reshape(-1)
    gy = gy.reshape(-1)

    pred_x = (win_probs * gx).sum()
    pred_y = (win_probs * gy).sum()

    return torch.stack([pred_x, pred_y]), ent


# ---------------------------------------------------------------------------
# Batched version (for use inside the model forward)
# ---------------------------------------------------------------------------

def batched_adaptive_softargmax(
    sim_rows: torch.Tensor,   # (B, N_kp, h*w)
    h: int,
    w: int,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:

    B, N_kp, _ = sim_rows.shape
    coords    = torch.zeros(B, N_kp, 2, device=sim_rows.device)
    entropies = torch.zeros(B, N_kp, device=sim_rows.device)

    for b in range(B):
        for k in range(N_kp):
            coord, ent = adaptive_window_softargmax(sim_rows[b, k], h, w, **kwargs)
            coords[b, k]    = coord
            entropies[b, k] = ent

    return coords, entropies
