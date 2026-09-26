"""NumPy reference TPU plus shared helpers (FP32 bit patterns, command counters)."""
from __future__ import annotations

from collections import Counter

import numpy as np

TILE = 4

# Command ids, used only as keys for the per-command counters in summary().
CMD_GEMM_CLR = 1
CMD_GEMM_ACC = 2
CMD_VADD = 3
CMD_SILU = 6
CMD_RMSNORM = 7
CMD_SOFTMAX = 8
CMD_ROPE = 9
CMD_ACT_LUT = 10
CMD_VMUL = 13


def to_f32(x):
    """IEEE-754 float32 bit patterns as signed int64 (safe to print / fscanf)."""
    return np.asarray(x, dtype=np.float32).view(np.int32).astype(np.int64)


def from_f32(q):
    return np.asarray(q, dtype=np.int32).view(np.float32).astype(np.float64)


def rope_lut_angles(n_pairs, position, theta=10000.0):
    """ABI v2: FP32 radians as signed bit patterns (legacy helper name)."""
    inv_freq = 1.0 / (float(theta) ** (np.arange(n_pairs, dtype=np.float64) / n_pairs))
    rad = float(position) * inv_freq
    return to_f32(rad)


def _pad4(n):
    return (n + TILE - 1) // TILE * TILE


class TpuLlm:
    """Pure-NumPy TPU: same commands and FP32 rounding as tpu_fpga.sv."""

    def __init__(self):
        self.counts = Counter()
        self.gemm_tiles = 0

    def reset_stats(self):
        self.counts.clear()
        self.gemm_tiles = 0

    def vadd(self, x, y):
        self.counts[CMD_VADD] += 1
        return (np.asarray(x, dtype=np.float32) + np.asarray(y, dtype=np.float32)).astype(np.float64)

    def vmul(self, x, scale):
        self.counts[CMD_VMUL] += 1
        return (np.asarray(x, dtype=np.float32) * np.float32(scale)).astype(np.float64)

    def vmul_elem(self, x, y):
        self.counts[CMD_VMUL] += 1
        return (np.asarray(x, dtype=np.float32) * np.asarray(y, dtype=np.float32)).astype(np.float64)

    def rmsnorm(self, x, gamma, eps=1e-5):
        self.counts[CMD_RMSNORM] += 1
        x = np.asarray(x, dtype=np.float32)
        gamma = np.asarray(gamma, dtype=np.float32)
        mean_sq = np.mean(x * x, axis=-1, keepdims=True)
        return (x * (1.0 / np.sqrt(mean_sq + np.float32(eps))) * gamma).astype(np.float64)

    def silu(self, x, use_lut=True):
        x = np.asarray(x, dtype=np.float32)
        if use_lut:
            self.counts[CMD_ACT_LUT] += 1
        else:
            self.counts[CMD_SILU] += 1
        xf = x.astype(np.float64)
        e = np.exp(-np.abs(xf))
        return np.where(xf >= 0, xf / (1 + e), xf * e / (1 + e)).astype(np.float32).astype(np.float64)

    def rope(self, x, position, theta=10000.0):
        """Pair-wise RoPE (same adjacent-pair layout as OP_ROPE)."""
        self.counts[CMD_ROPE] += 1
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        half = x.size // 2
        inv_freq = 1.0 / (float(theta) ** (np.arange(half, dtype=np.float64) / half))
        ang = (float(position) * inv_freq).astype(np.float32).astype(np.float64)
        c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
        y = np.empty_like(x, dtype=np.float64)
        even, odd = x[0::2], x[1::2]
        y[0::2] = even * c - odd * s
        y[1::2] = even * s + odd * c
        return y

    def softmax(self, logits, valid_len=None):
        self.counts[CMD_SOFTMAX] += 1
        z = np.asarray(logits, dtype=np.float32)
        if z.size == 0:
            return np.zeros(0, dtype=np.float64)
        if valid_len == 0:
            valid_len = None  # ISA: zero means all lanes valid
        if valid_len is not None and not 0 < valid_len <= z.size:
            raise ValueError("valid_len must be between 0 and vector length")
        if valid_len is not None:
            z = z[:valid_len]
        z = z - np.max(z)
        e = np.exp(z.astype(np.float64))
        p = (e / np.sum(e)).astype(np.float32).astype(np.float64)
        if valid_len is not None:
            out = np.zeros(len(logits), dtype=np.float64)
            out[:valid_len] = p
            return out
        return p

    def linear(self, x, weight, weight_scale=None, bias=None):
        """y = W x + b in FP32."""
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        w = np.asarray(weight, dtype=np.float32)
        y = self._tiled_gemv(w, x).astype(np.float64)
        if bias is not None:
            y = self.vadd(y, bias)
        return y

    def linear_rows(self, xs, weight, weight_scale=None, bias=None):
        return np.stack([self.linear(row, weight, weight_scale, bias) for row in xs], axis=0)

    def _tiled_gemv(self, w, x):
        m, k = w.shape
        rows, cols = _pad4(m) // TILE, _pad4(k) // TILE
        tiles = rows * cols
        self.gemm_tiles += tiles
        self.counts[CMD_GEMM_CLR] += rows
        self.counts[CMD_GEMM_ACC] += max(tiles - rows, 0)
        return np.asarray(w, dtype=np.float32) @ np.asarray(x, dtype=np.float32)

    def summary(self):
        names = {
            CMD_GEMM_CLR: "GEMM_CLR",
            CMD_GEMM_ACC: "GEMM_ACC",
            CMD_VADD: "VADD",
            CMD_SOFTMAX: "SOFTMAX",
            CMD_SILU: "SILU",
            CMD_ACT_LUT: "ACT_LUT",
            CMD_RMSNORM: "RMSNORM",
            CMD_ROPE: "ROPE",
            CMD_VMUL: "VMUL",
        }
        parts = [f"{names.get(k, k)}={v}" for k, v in sorted(self.counts.items())]
        return f"tiles={self.gemm_tiles} " + " ".join(parts)
