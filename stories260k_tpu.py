"""stories260K Llama forward pass issued as TPU commands."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tpu_llm_sim import TpuLlm


@dataclass
class Stories260KConfig:
    n_layer: int = 5
    n_head: int = 8
    n_kv_head: int = 4
    n_embd: int = 64
    intermediate_size: int = 172
    block_size: int = 128
    vocab_size: int = 512
    rms_eps: float = 1e-5
    rope_theta: float = 10000.0


def _as32(x, default_shape=None, default=0.0):
    if x is None:
        return np.zeros(default_shape, dtype=np.float32) + default
    return np.asarray(x, dtype=np.float32)


class Stories260KTPU:
    def __init__(self, cfg: Stories260KConfig, weights: dict, tpu=None, silu_lut=True):
        self.cfg = cfg
        self.tpu = tpu or TpuLlm()
        self.silu_lut = silu_lut
        if min(cfg.n_layer, cfg.n_head, cfg.n_kv_head, cfg.n_embd, cfg.intermediate_size, cfg.block_size, cfg.vocab_size) <= 0:
            raise ValueError("model dimensions must be positive")
        if cfg.n_embd % cfg.n_head:
            raise ValueError("n_head must divide n_embd")
        if cfg.n_head % cfg.n_kv_head:
            raise ValueError("n_kv_head must divide n_head")
        self.head_dim = cfg.n_embd // cfg.n_head
        if self.head_dim % 2:
            raise ValueError("RoPE head dimension must be even")
        vmax = getattr(self.tpu, "vlen", 2048)
        if self.head_dim > vmax or cfg.n_embd > max(vmax, 2048) or cfg.block_size > max(vmax, 2048):
            raise ValueError("model dimensions exceed TPU RoPE/reduction capacity")
        self.n_rep = cfg.n_head // cfg.n_kv_head
        self.attn_scale = 1.0 / math.sqrt(self.head_dim)
        self.wte = _as32(weights["wte"])
        self.lm_w = _as32(weights.get("lm_head", weights["wte"]))
        self.norm_w = _as32(weights["norm_w"])
        self.layers = [self._prep_layer(weights, i) for i in range(cfg.n_layer)]
        shapes = {"wte": (cfg.vocab_size, cfg.n_embd), "norm_w": (cfg.n_embd,)}
        kv_dim = cfg.n_kv_head * self.head_dim
        for i in range(cfg.n_layer):
            for key, shape in {"attn_norm": (cfg.n_embd,), "ffn_norm": (cfg.n_embd,),
                               "q": (cfg.n_embd, cfg.n_embd), "k": (kv_dim, cfg.n_embd),
                               "v": (kv_dim, cfg.n_embd), "o": (cfg.n_embd, cfg.n_embd),
                               "gate": (cfg.intermediate_size, cfg.n_embd),
                               "up": (cfg.intermediate_size, cfg.n_embd),
                               "down": (cfg.n_embd, cfg.intermediate_size)}.items():
                shapes[f"h.{i}.{key}.w"] = shape
        for key, shape in shapes.items():
            if np.asarray(weights[key]).shape != shape or not np.all(np.isfinite(weights[key])):
                raise ValueError(f"invalid tensor {key}; expected finite values and shape {shape}")
        if self.lm_w.shape != (cfg.vocab_size, cfg.n_embd) or not np.all(np.isfinite(self.lm_w)):
            raise ValueError("invalid output weight")

    def _prep_layer(self, weights, i):
        def pack(key):
            w = _as32(weights[f"h.{i}.{key}.w"])
            b = weights.get(f"h.{i}.{key}.b")
            b = None if b is None else _as32(b)
            return w, None, b

        return {
            "attn_norm": _as32(weights[f"h.{i}.attn_norm.w"]),
            "ffn_norm": _as32(weights[f"h.{i}.ffn_norm.w"]),
            "q": pack("q"),
            "k": pack("k"),
            "v": pack("v"),
            "o": pack("o"),
            "gate": pack("gate"),
            "up": pack("up"),
            "down": pack("down"),
        }

    def _rms_rows(self, x, gamma):
        return np.stack(
            [self.tpu.rmsnorm(row, gamma, eps=self.cfg.rms_eps) for row in x],
            axis=0,
        )

    def _rope_heads(self, x, pos_offset):
        t, n_h, _ = x.shape
        out = np.empty_like(x)
        for ti in range(t):
            pos = pos_offset + ti
            for h in range(n_h):
                out[ti, h] = self.tpu.rope(x[ti, h], pos, self.cfg.rope_theta)
        return out

    def _attention(self, x, layer, pos_offset, past_kv=None):
        tpu, cfg = self.tpu, self.cfg
        t = x.shape[0]
        q = tpu.linear_rows(x, *layer["q"]).reshape(t, cfg.n_head, self.head_dim)
        k = tpu.linear_rows(x, *layer["k"]).reshape(t, cfg.n_kv_head, self.head_dim)
        v = tpu.linear_rows(x, *layer["v"]).reshape(t, cfg.n_kv_head, self.head_dim)
        q = self._rope_heads(q, pos_offset)
        k = self._rope_heads(k, pos_offset)
        if past_kv is not None:
            k = np.concatenate([past_kv[0], k], axis=0)
            v = np.concatenate([past_kv[1], v], axis=0)
        y = np.zeros((t, cfg.n_head, self.head_dim), dtype=np.float64)
        for h in range(cfg.n_head):
            hkv = h // self.n_rep
            for ti in range(t):
                abs_t = pos_offset + ti
                kt = k[: abs_t + 1, hkv]
                scores = tpu.vmul(tpu.linear(q[ti, h], kt), self.attn_scale)
                probs = tpu.softmax(scores, valid_len=abs_t + 1)[: abs_t + 1]
                y[ti, h] = tpu.linear(probs, v[: abs_t + 1, hkv].T)
        out = tpu.linear_rows(y.reshape(t, cfg.n_embd), *layer["o"])
        return out, (k, v)

    def _mlp(self, x, layer):
        gate = self.tpu.silu(self.tpu.linear_rows(x, *layer["gate"]), use_lut=self.silu_lut)
        up = self.tpu.linear_rows(x, *layer["up"])
        hidden = self.tpu.vmul_elem(gate, up)
        return self.tpu.linear_rows(hidden, *layer["down"])

    def forward(self, ids, past_kv=None):
        ids = list(ids)
        if not ids or any(not isinstance(i, (int, np.integer)) or not 0 <= i < self.cfg.vocab_size for i in ids):
            raise ValueError("ids must contain valid integer token IDs")
        if past_kv is None:
            pos_offset = 0
        else:
            if len(past_kv) != self.cfg.n_layer:
                raise ValueError("KV cache layer count mismatch")
            used = past_kv[0][0].shape[0]
            expected = (used, self.cfg.n_kv_head, self.head_dim)
            if any(k.shape != expected or v.shape != expected for k, v in past_kv):
                raise ValueError("KV cache shape mismatch")
            pos_offset = used
        if pos_offset + len(ids) > self.cfg.block_size:
            raise ValueError("context window exceeded; reset the cache or shorten input")
        x = self.wte[np.asarray(ids, dtype=np.int64)].astype(np.float64)
        new_kv = []
        for i, layer in enumerate(self.layers):
            pk = None if past_kv is None else past_kv[i]
            attn, kv = self._attention(
                self._rms_rows(x, layer["attn_norm"]), layer, pos_offset, pk
            )
            x = np.stack([self.tpu.vadd(x[j], attn[j]) for j in range(len(x))], axis=0)
            mlp = self._mlp(self._rms_rows(x, layer["ffn_norm"]), layer)
            x = np.stack([self.tpu.vadd(x[j], mlp[j]) for j in range(len(x))], axis=0)
            new_kv.append(kv)
        xf = self.tpu.rmsnorm(x[-1], self.norm_w, eps=self.cfg.rms_eps)
        logits = self.tpu.linear(xf, self.lm_w, bias=None)
        if not np.all(np.isfinite(logits)):
            raise FloatingPointError("non-finite logits produced during inference")
        return logits, new_kv

    def generate(self, ids, max_new_tokens, eos=None):
        ids = list(ids)
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        if not ids or len(ids) + max(0, max_new_tokens - 1) > self.cfg.block_size:
            raise ValueError("prompt plus requested generation exceeds context window")
        past_kv = None
        for step in range(max_new_tokens):
            step_ids = ids if past_kv is None else [ids[-1]]
            logits, past_kv = self.forward(step_ids, past_kv=past_kv)
            nxt = int(np.argmax(logits))
            ids.append(nxt)
            print(f"  token {step + 1}/{max_new_tokens} -> {nxt}", flush=True)
            if eos is not None and nxt == eos:
                break
        return ids
