"""Run stories260K.gguf on the clocked TPU (tpu_fpga.sv) or the NumPy reference."""
from __future__ import annotations

import argparse
import re
import struct
import sys
from pathlib import Path

HOST = Path(__file__).resolve().parent
sys.path.insert(0, str(HOST))

from stories260k_tpu import Stories260KConfig, Stories260KTPU
from tpu_fpga import TpuLlmFpga
from tpu_llm_sim import TpuLlm

DEFAULT_GGUF = HOST / "stories260K.gguf"
_BYTE_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def _u32(data, off):
    return struct.unpack_from("<I", data, off)[0], off + 4


def _u64(data, off):
    return struct.unpack_from("<Q", data, off)[0], off + 8


def _str(data, off):
    n, off = _u64(data, off)
    s = data[off:off + n].decode("utf-8", "replace")
    return s, off + n


def _val(data, off, typ=None):
    if typ is None:
        typ, off = _u32(data, off)
    if typ == 8:
        return _str(data, off)
    if typ == 7:
        return bool(data[off]), off + 1
    fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 10: "<Q", 11: "<q", 12: "<d"}
    if typ in fmt:
        v = struct.unpack_from(fmt[typ], data, off)[0]
        return v, off + struct.calcsize(fmt[typ])
    if typ == 9:
        et, off = _u32(data, off)
        n, off = _u64(data, off)
        items = []
        for _ in range(int(n)):
            v, off = _val(data, off, et)
            items.append(v)
        return items, off
    raise ValueError(f"unsupported GGUF type {typ}")


def parse_gguf(path: str | Path):
    import numpy as np

    data = Path(path).read_bytes()
    if data[:4] != b"GGUF":
        raise ValueError(f"{path} is not a GGUF file")
    off = 4
    ver, off = _u32(data, off)
    if ver not in (2, 3):
        raise ValueError(f"unsupported GGUF version {ver}")
    n_tensors, off = _u64(data, off)
    n_kv, off = _u64(data, off)
    meta = {}
    for _ in range(int(n_kv)):
        k, off = _str(data, off)
        v, off = _val(data, off)
        meta[k] = v
    infos = []
    for _ in range(int(n_tensors)):
        name, off = _str(data, off)
        nd, off = _u32(data, off)
        dims = []
        for _i in range(nd):
            d, off = _u64(data, off)
            dims.append(int(d))
        dtype, off = _u32(data, off)
        toff, off = _u64(data, off)
        infos.append((name, dims, dtype, int(toff)))
    align = int(meta.get("general.alignment", 32))
    if align <= 0 or align & (align - 1):
        raise ValueError("GGUF alignment must be a positive power of two")
    data_base = (off + align - 1) // align * align
    tensors = {}
    for name, dims, dtype, toff in infos:
        if name in tensors or not dims or any(d <= 0 for d in dims):
            raise ValueError(f"invalid/duplicate tensor {name}")
        if dtype != 0:
            raise ValueError(f"{name}: only F32 GGUF tensors are supported (got {dtype})")
        n = 1
        for d in dims:
            n *= d
        raw = data[data_base + toff:data_base + toff + n * 4]
        if toff % align or len(raw) != n * 4:
            raise ValueError(f"{name}: truncated or misaligned tensor data")
        arr = np.frombuffer(raw, dtype="<f4").reshape(dims[::-1]).astype(np.float32)
        tensors[name] = arr
    return meta, tensors


class GgufTokenizer:
    def __init__(self, tokens, bos=1, eos=2, unk=0, scores=None, add_bos=True, add_space_prefix=True):
        self.tokens = list(tokens)
        self.bos_token_id = int(bos)
        self.eos_token_id = int(eos)
        self.unk_token_id = int(unk)
        self.stoi = {t: i for i, t in enumerate(self.tokens)}
        self.scores = list(scores) if scores is not None else [0.0] * len(tokens)
        if len(self.scores) != len(self.tokens):
            raise ValueError("token scores and vocabulary lengths differ")
        self.add_bos = add_bos
        self.add_space_prefix = add_space_prefix
        self.byte_id = {}
        for i, t in enumerate(self.tokens):
            m = _BYTE_RE.match(t)
            if m:
                self.byte_id[int(m.group(1), 16)] = i
        self._max = max((len(t) for t in self.tokens), default=1)

    def encode(self, text):
        s = ((" " if text and self.add_space_prefix else "") + text).replace(" ", "\u2581")
        pieces = list(s)
        # SentencePiece BPE: repeatedly merge the highest-score adjacent pair,
        # resolving ties left-to-right. Greedy longest-prefix matching is wrong.
        while len(pieces) > 1:
            candidates = [(self.scores[self.stoi[a+b]], -i, i)
                          for i, (a, b) in enumerate(zip(pieces, pieces[1:])) if a+b in self.stoi]
            if not candidates:
                break
            _, _, i = max(candidates)
            pieces[i:i+2] = [pieces[i] + pieces[i+1]]
        ids = [self.bos_token_id] if self.add_bos else []
        for piece in pieces:
            if piece in self.stoi:
                ids.append(self.stoi[piece])
            else:
                ids.extend(self.byte_id.get(b, self.unk_token_id) for b in piece.encode("utf-8"))
        return ids

    def decode(self, ids, skip_special_tokens=True):
        skip = {self.bos_token_id, self.eos_token_id} if skip_special_tokens else set()
        out = bytearray()
        for i in ids:
            i = int(i)
            if i in skip or i < 0 or i >= len(self.tokens):
                continue
            t = self.tokens[i]
            m = _BYTE_RE.match(t)
            if m:
                out.append(int(m.group(1), 16))
            else:
                out.extend(t.replace("\u2581", " ").encode("utf-8"))
        text = out.decode("utf-8", "replace")
        if self.add_space_prefix and text.startswith(" "):
            text = text[1:]
        return text


def load_gguf_llama(path: str | Path):
    path = Path(path)
    print(f"loading GGUF {path} ...", flush=True)
    meta, t = parse_gguf(path)
    arch = meta.get("general.architecture", "llama")
    if arch != "llama":
        raise ValueError(f"expected llama GGUF, got {arch}")
    n_embd = int(meta["llama.embedding_length"])
    n_head = int(meta["llama.attention.head_count"])
    if meta.get("tokenizer.ggml.model") != "llama":
        raise ValueError("only the llama SentencePiece BPE tokenizer is supported")
    if n_head <= 0 or int(meta.get("llama.rope.dimension_count", n_embd // n_head)) != n_embd // n_head:
        raise ValueError("only full-head RoPE is supported")
    if meta.get("llama.rope.scaling.type", "none") != "none":
        raise ValueError("scaled RoPE is not supported")
    cfg = Stories260KConfig(
        n_layer=int(meta["llama.block_count"]),
        n_head=n_head,
        n_kv_head=int(meta.get("llama.attention.head_count_kv", n_head)),
        n_embd=n_embd,
        intermediate_size=int(meta["llama.feed_forward_length"]),
        block_size=int(meta.get("llama.context_length", 512)),
        vocab_size=int(t["token_embd.weight"].shape[0]),
        rms_eps=float(meta.get("llama.attention.layer_norm_rms_epsilon", 1e-5)),
        rope_theta=float(meta.get("llama.rope.freq_base", 10000.0)),
    )
    w = {
        "wte": t["token_embd.weight"],
        "lm_head": t.get("output.weight", t["token_embd.weight"]),
        "norm_w": t["output_norm.weight"],
    }
    for i in range(cfg.n_layer):
        p = f"blk.{i}"
        w[f"h.{i}.attn_norm.w"] = t[f"{p}.attn_norm.weight"]
        w[f"h.{i}.ffn_norm.w"] = t[f"{p}.ffn_norm.weight"]
        w[f"h.{i}.q.w"] = t[f"{p}.attn_q.weight"]
        w[f"h.{i}.k.w"] = t[f"{p}.attn_k.weight"]
        w[f"h.{i}.v.w"] = t[f"{p}.attn_v.weight"]
        w[f"h.{i}.o.w"] = t[f"{p}.attn_output.weight"]
        w[f"h.{i}.gate.w"] = t[f"{p}.ffn_gate.weight"]
        w[f"h.{i}.up.w"] = t[f"{p}.ffn_up.weight"]
        w[f"h.{i}.down.w"] = t[f"{p}.ffn_down.weight"]
    expected = {"token_embd.weight", "output.weight", "output_norm.weight"}
    expected.update(f"blk.{i}.{name}.weight" for i in range(cfg.n_layer)
                    for name in ("attn_norm", "ffn_norm", "attn_q", "attn_k", "attn_v",
                                 "attn_output", "ffn_gate", "ffn_up", "ffn_down"))
    if set(t) - expected:
        raise ValueError(f"unsupported model tensors: {sorted(set(t) - expected)}")
    if len(meta["tokenizer.ggml.tokens"]) != cfg.vocab_size:
        raise ValueError("tokenizer vocabulary does not match embedding rows")
    tok = GgufTokenizer(
        meta["tokenizer.ggml.tokens"],
        bos=int(meta.get("tokenizer.ggml.bos_token_id", 1)),
        eos=int(meta.get("tokenizer.ggml.eos_token_id", 2)),
        unk=int(meta.get("tokenizer.ggml.unknown_token_id", 0)),
        scores=meta["tokenizer.ggml.scores"],
        add_bos=bool(meta.get("tokenizer.ggml.add_bos_token", True)),
        add_space_prefix=bool(meta.get("tokenizer.ggml.add_space_prefix", True)),
    )
    return cfg, w, tok


def main():
    p = argparse.ArgumentParser(description="stories260K on the clocked TPU or the NumPy reference")
    p.add_argument("--gguf", default=str(DEFAULT_GGUF), help="path to .gguf")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--tool-dir", help="directory containing iverilog and vvp")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument(
        "--backend",
        choices=("fpga", "numpy"),
        default="fpga",
        help="fpga = clocked RTL in Icarus; numpy = Python reference",
    )
    args = p.parse_args()

    cfg, weights, tok = load_gguf_llama(args.gguf)
    print(
        f"{Path(args.gguf).name} FP32 ({args.backend}): layers={cfg.n_layer} "
        f"heads={cfg.n_head} kv={cfg.n_kv_head} embd={cfg.n_embd} "
        f"ffn={cfg.intermediate_size} vocab={cfg.vocab_size}",
        flush=True,
    )
    if args.backend == "numpy":
        tpu = TpuLlm()
    else:
        tpu = TpuLlmFpga(tool_dir=args.tool_dir)
    model = Stories260KTPU(cfg, weights, tpu=tpu)
    ids = tok.encode(args.prompt)
    print("prompt:", args.prompt, flush=True)
    print("prompt ids:", ids, flush=True)
    try:
        out = model.generate(ids, args.max_new_tokens, eos=tok.eos_token_id)
    finally:
        if hasattr(tpu, "close"):
            tpu.close()
    print("generated ids:", out[len(ids):], flush=True)
    print("text:", tok.decode(out), flush=True)
    print("tpu cmds:", tpu.summary(), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
