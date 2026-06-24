"""
order_tracking.py
-----------------
Differentiable order-domain resampling layer.

A bearing fault's characteristic frequency lives at  k × f_r (k = order),
where f_r is the shaft rotation frequency. If we resample the vibration
signal so that the new axis is "shaft revolution count" rather than time,
the fault peaks always appear at the SAME order regardless of shaft speed.

This layer takes:
   - raw time-domain window of length WINDOW_SAMPLES  (e.g. 4096)
   - shaft RPM and sampling rate fs (both per-sample in batch)
and returns:
   - magnitude spectrum in the ORDER domain, shape (B, ORDER_BINS)
   - each bin corresponds to a fraction of shaft order (0 .. MAX_ORDER)

This is differentiable end-to-end. We rely on torch.fft.rfft.

Why this matters for the paper:
  - Without this layer, training data from CWRU (~1772 rpm) and HUST (~700 rpm)
    look completely different to a CNN — the SAME fault produces peaks at
    different absolute frequencies.
  - With this layer, all signals share a common geometry-aware coordinate
    system. This is the "geometry-invariant front end" Pillar 1 from the
    proposal.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ORDER_BINS, MAX_ORDER, WINDOW_SAMPLES


class OrderTrackingLayer(nn.Module):
    """
    Convert a batch of time-domain windows → order-domain magnitude spectra.

    Math:
      - shaft rotation frequency  f_r [Hz] = RPM / 60
      - the FFT of a window of length L samples at fs Hz has frequency bins
        f_k = k * fs / L  for k = 0..L/2.
      - to express in shaft orders: order_k = f_k / f_r = k * fs / (L * f_r).
      - we resample the magnitude spectrum onto a fixed grid of order bins
        [0, MAX_ORDER] with ORDER_BINS points.

    Output is (B, 1, ORDER_BINS) — keep the channel dim for downstream CNN.
    """

    def __init__(self, n_order_bins=ORDER_BINS, max_order=MAX_ORDER):
        super().__init__()
        self.n_order_bins = n_order_bins
        self.max_order = max_order
        # Fixed query grid of orders we want to evaluate
        self.register_buffer(
            "order_grid",
            torch.linspace(0.0, max_order, n_order_bins),
        )

    def forward(self, x, fs, rpm):
        """
        x:   (B, 1, L)   time-domain windows
        fs:  (B,)        Hz
        rpm: (B,)        rpm
        Returns: (B, 1, n_order_bins) magnitude in order domain.
        """
        B, C, L = x.shape
        assert C == 1, "OrderTrackingLayer expects single-channel input"

        # Light Hann window to suppress leakage
        win = torch.hann_window(L, device=x.device, dtype=x.dtype)
        xw = x[:, 0, :] * win                     # (B, L)

        # Real FFT — only L/2+1 unique bins
        X = torch.fft.rfft(xw, n=L)               # (B, L//2+1)
        mag = X.abs()                             # (B, K) where K=L//2+1
        K = mag.shape[1]

        # Each FFT bin k corresponds to frequency f_k = k * fs / L (Hz),
        # and to order o_k = f_k / f_r where f_r = rpm/60.
        # For per-sample fs and rpm, build per-sample "order coordinate" of bins.
        f_r = (rpm / 60.0).clamp(min=1e-3)        # (B,) Hz
        # bin → freq scale per sample
        df = fs / L                               # (B,)
        # max order present in the FFT per sample
        # For interpolation we need a normalized index into mag.

        # We want, for each query order o_q in self.order_grid, the bin index
        # k_q = o_q * f_r / df = o_q * f_r * L / fs
        # then we interpolate mag[k_q].

        # Compute query indices: shape (B, n_order_bins)
        # Each sample has its own scaling.
        scale = (f_r * L / fs).unsqueeze(1)        # (B, 1) — bins per order
        query_idx = self.order_grid.unsqueeze(0) * scale   # (B, n_order_bins)

        # Linear interpolation along the bin axis.
        # We use grid_sample for simplicity & differentiability.
        # mag shape (B, K) → reshape to (B, 1, 1, K)
        mag_4d = mag.unsqueeze(1).unsqueeze(1)              # (B,1,1,K)

        # grid_sample requires grid in normalized coords [-1, 1].
        # Map query_idx ∈ [0, K-1] → [-1, 1].
        grid_x = (query_idx / max(K - 1, 1)) * 2.0 - 1.0    # (B, n_order_bins)
        # Out-of-range orders (where order > max available) get clamped — we mask
        # those out at the end.
        grid_x_clamped = grid_x.clamp(-1.0, 1.0)
        # build (B, 1, n_order_bins, 2) grid: y=0 (only 1 row), x=grid_x
        zero = torch.zeros_like(grid_x_clamped)
        grid = torch.stack([grid_x_clamped, zero], dim=-1).unsqueeze(1)  # (B,1,n_order,2)

        order_spec = F.grid_sample(
            mag_4d, grid, mode="bilinear", padding_mode="zeros",
            align_corners=True,
        )    # (B, 1, 1, n_order_bins)
        order_spec = order_spec.squeeze(2)                 # (B, 1, n_order_bins)

        # Mask out queries that fell outside [0, K-1] (i.e. asking for orders
        # beyond the Nyquist available in this sample)
        mask = ((query_idx >= 0) & (query_idx <= K - 1)).float().unsqueeze(1)
        order_spec = order_spec * mask

        # Log-magnitude (compresses dynamic range, common in fault diagnosis)
        order_spec = torch.log1p(order_spec)

        # Per-sample normalisation
        mu = order_spec.mean(dim=-1, keepdim=True)
        sd = order_spec.std(dim=-1, keepdim=True) + 1e-6
        order_spec = (order_spec - mu) / sd

        return order_spec


# Quick self-test
if __name__ == "__main__":
    layer = OrderTrackingLayer()
    x = torch.randn(4, 1, WINDOW_SAMPLES)
    fs = torch.tensor([12000.0, 24000.0, 64000.0, 97656.0])
    rpm = torch.tensor([1772.0, 700.0, 900.0, 1500.0])
    out = layer(x, fs, rpm)
    print("input :", x.shape)
    print("output:", out.shape, "min", out.min().item(), "max", out.max().item())
