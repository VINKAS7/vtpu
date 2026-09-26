"""Drive the clocked FPGA TPU (tpu_fpga.sv) through Icarus."""
from __future__ import annotations

import atexit
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from tpu_llm_sim import (
    CMD_ACT_LUT,
    CMD_GEMM_ACC,
    CMD_GEMM_CLR,
    CMD_RMSNORM,
    CMD_ROPE,
    CMD_SILU,
    CMD_SOFTMAX,
    CMD_VADD,
    CMD_VMUL,
    TpuLlm,
    from_f32,
    rope_lut_angles,
    to_f32,
)

HOST = Path(__file__).resolve().parent
SV = HOST / "tpu_fpga.sv"
TB = HOST / "tb_tpu_fpga.sv"

# Sizes, opcodes and memory selects come from hw_params.svh, the table the RTL uses too.
HW = {k: int(v) for k, v in re.findall(r"`define\s+(\w+)\s+(\d+)", (HOST / "hw_params.svh").read_text())}
(OP_WR, OP_RD, OP_GEMV, OP_VADD, OP_VMUL, OP_VMUL_ELEM, OP_SILU, OP_RMSNORM, OP_SOFTMAX, OP_ROPE,
 OP_QUIT, OP_CYCLES) = (HW["OP_" + n] for n in ("WR", "RD", "GEMV", "VADD", "VMUL", "VMUL_ELEM", "SILU",
                                               "RMSNORM", "SOFTMAX", "ROPE", "QUIT", "CYCLES"))
CLOCK_HZ = 100e6  # tb_tpu_fpga.sv's nominal clock (a 10 ns period); the core isn't synthesized yet
SEL_W, SEL_X, SEL_Y, SEL_G, SEL_A = (HW["SEL_" + n] for n in "WXYGA")
VLEN = HW["VLEN"]
WMEM_WORDS = HW["WMEM_WORDS"]
SCRATCH = WMEM_WORDS - HW["SCRATCH_WORDS"]
MEM_WORDS = {SEL_W: WMEM_WORDS, SEL_X: HW["XMEM_WORDS"], SEL_Y: HW["YMEM_WORDS"], SEL_G: VLEN, SEL_A: VLEN // 2}
OP_NAMES = {v: k[3:] for k, v in HW.items() if k.startswith("OP_")}
WR_CHUNK = 2048

# Seconds a transaction may take: a fixed allowance, plus a generous cost per simulated cycle and
# per word written. A hung simulator is reported in seconds, not minutes.
TXN_BASE_S = 20.0
TXN_S_PER_CYCLE = 2e-4
TXN_S_PER_WORD = 2e-4

# Where Icarus usually lives on Windows, when it isn't on the PATH.
_ICARUS_HINTS = [
    str(HOST.parent / "iverilog" / "app" / "bin"),
    r"C:\iverilog\bin",
    r"C:\oss-cad-suite\bin",
    r"C:\Program Files\iverilog\bin",
    r"C:\Program Files (x86)\iverilog\bin",
]


def _unlink_flag(path):
    deadline = time.monotonic() + 5
    while True:
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.002)


def _need(bin_name: str) -> str:
    path = shutil.which(bin_name)
    if path:
        return path
    if os.name == "nt":
        for folder in _ICARUS_HINTS:
            candidate = Path(folder) / f"{bin_name}.exe"
            if candidate.is_file():
                return str(candidate)
    raise RuntimeError(
        f"{bin_name} not found on PATH. Install Icarus Verilog:\n"
        "  https://bleyer.org/icarus/"
    )


class TpuLlmFpga(TpuLlm):
    """Same TpuLlm API; weights stay in on-chip SRAM, one command per GEMV/vector op."""

    def __init__(self, tool_dir=None):
        super().__init__()
        self.vlen = VLEN
        self.tool_dir = Path(tool_dir) if tool_dir else None
        self._temp = tempfile.TemporaryDirectory(prefix="tinytpu-fpga-")
        self.work = Path(self._temp.name)
        self._go = self.work / "go"
        self._done = self.work / "done"
        self._ready = self.work / "ready"
        self._cmd = self.work / "cmd.txt"
        self._rsp = self.work / "rsp.txt"
        self._log = self.work / "vvp.log"
        self._proc = None
        self._heap = 0
        self._wmap = {}
        self._pending = []  # queued write commands, sent with the next transaction
        self._txns = 0
        self._last = None
        try:
            self._compile()
            self._start()
        except Exception:
            self.close()
            raise
        atexit.register(self.close)

    def _compile(self):
        suffix = ".exe" if os.name == "nt" else ""
        iverilog = str(self.tool_dir / ("iverilog" + suffix)) if self.tool_dir else _need("iverilog")
        self.vvp_bin = str(self.tool_dir / ("vvp" + suffix)) if self.tool_dir else _need("vvp")
        out = self.work / "tpu_fpga.vvp"
        lib = Path(iverilog).parent.parent / "lib" / "ivl"
        self._lib = lib if lib.is_dir() else None
        print(f"compiling tpu_fpga.sv (1 MAC/cycle GEMV, VLEN={VLEN}, on-chip weights) ...", flush=True)
        proc = subprocess.run(
            [
                iverilog,
                *(["-B", str(self._lib)] if self._lib else []),
                "-g2012",
                "-I",
                str(HOST),
                "-s",
                "tb_tpu_fpga",
                "-o",
                str(out),
                str(SV),
                str(TB),
            ],
            capture_output=True,
            text=True,
            cwd=str(HOST),
        )
        if proc.returncode != 0:
            raise RuntimeError("iverilog failed:\n" + (proc.stdout or "") + (proc.stderr or ""))
        self.vvp_file = out
        print(f"compiled {out.name}", flush=True)

    def _plus(self, key, path: Path) -> str:
        return f"+{key}={path.resolve().as_posix()}"

    def _start(self):
        for p in (self._go, self._done, self._ready, self._cmd, self._rsp):
            if p.exists():
                _unlink_flag(p)
        logf = open(self._log, "w", encoding="utf-8")
        self._logf = logf
        self._proc = subprocess.Popen(
            [
                self.vvp_bin,
                *(["-M", "-", "-M", str(self._lib)] if self._lib else []),
                str(self.vvp_file.resolve()),
                self._plus("READY", self._ready),
                self._plus("GO", self._go),
                self._plus("DONE", self._done),
                self._plus("CMD", self._cmd),
                self._plus("RSP", self._rsp),
            ],
            cwd=str(self.work),
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
        t0 = time.time()
        while not self._ready.exists():
            if self._proc.poll() is not None:
                logf.flush()
                tail = self._log.read_text(encoding="utf-8", errors="replace")[-2000:]
                raise RuntimeError("vvp exited before READY:\n" + tail)
            if time.time() - t0 > 120:
                logf.flush()
                tail = self._log.read_text(encoding="utf-8", errors="replace")[-2000:]
                raise TimeoutError("vvp did not create READY. log:\n" + tail)
            time.sleep(0.05)
        print(f"tpu_fpga.sv running via vvp  (pid={self._proc.pid})", flush=True)

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        if self._proc is not None and self._proc.poll() is None:
            if self._ready.exists() and not getattr(self, "_failed", False):  # TB idle; otherwise just kill it
                self._pending = []
                try:
                    self._txn([OP_QUIT], expect=0, timeout=5, quit=True)
                except Exception:
                    pass
            try:
                self._proc.kill()
                self._proc.wait(timeout=5)
            except Exception:
                pass
        self._proc = None
        logf = getattr(self, "_logf", None)
        if logf:
            try:
                logf.close()
            except Exception:
                pass
            self._logf = None
        try:
            self._temp.cleanup()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _write_cmd(self, lines):
        with open(self._cmd, "w", encoding="ascii", newline="\n") as f:
            for parts in lines:
                f.write(" ".join(str(int(x)) for x in parts) + "\n")
            f.flush()
            os.fsync(f.fileno())

    # Strict 4-phase handshake with tb_tpu_fpga.sv:
    #   READY present -> host deletes READY, writes CMD, creates GO
    #   TB runs, creates DONE -> host reads RSP, deletes GO
    #   TB sees GO gone, creates READY -> host deletes DONE
    # Every transaction (except QUIT) waits for READY before returning, and a
    # new one starts only once READY exists. Recreating GO before the TB has
    # observed it gone deadlocks both sides forever.
    def _fail(self, exc_type, what):
        """Stop at once, with what the host was doing and the state of both sides, and kill vvp."""
        flags = " ".join(f"{p.name}={'yes' if p.exists() else 'no'}" for p in (self._go, self._done, self._ready))
        tail = ""
        try:
            self._logf.flush()
            tail = self._log.read_text(encoding="utf-8", errors="replace")[-2000:]
        except Exception:
            pass
        cmd = "none" if self._last is None else f"{OP_NAMES.get(self._last[0], self._last[0])} {list(self._last[1:4])}"
        self._failed = True
        msg = (f"TPU {what}\n  transaction #{self._txns}, last command: {cmd}\n  flag files: {flags}\n"
               f"  vvp log tail:\n{tail}")
        self.close()
        raise exc_type(msg)

    def _wait_flag(self, path, timeout, what):
        t0 = time.monotonic()
        while not path.exists():
            if self._proc.poll() is not None:
                self._fail(RuntimeError, f"vvp exited while waiting for {what}")
            if time.monotonic() - t0 > timeout:
                self._fail(TimeoutError, f"{what} timed out after {timeout:.0f}s")
            time.sleep(0.002)

    def _begin_txn(self):
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("vvp is not running — FPGA TPU sim died")
        self._wait_flag(self._ready, TXN_BASE_S, "READY (previous command not acknowledged)")
        _unlink_flag(self._ready)

    def _end_txn(self, timeout, quit=False):
        self._go.write_text("1", encoding="ascii")
        self._wait_flag(self._done, timeout, "command")
        raw = self._rsp.read_text(encoding="ascii").split()
        _unlink_flag(self._go)
        if not quit:
            self._wait_flag(self._ready, TXN_BASE_S, "acknowledgement of GO removal")
        _unlink_flag(self._done)
        return [int(x) for x in raw]

    def _txn(self, parts, expect=None, timeout=None, quit=False, cycles=0):
        """Send the queued writes and one command in a single transaction; return its values."""
        lines = self._pending + [parts]
        words = sum(len(l) for l in lines)
        self._pending = []
        self._last = parts
        self._txns += 1
        self._begin_txn()
        self._write_cmd(lines)
        if timeout is None:
            timeout = TXN_BASE_S + cycles * TXN_S_PER_CYCLE + words * TXN_S_PER_WORD
        vals = self._end_txn(timeout, quit=quit)
        # One answer per command: a count and that many values, or -1 if it was refused.
        pos = 0
        for line in lines:
            if pos >= len(vals):
                raise RuntimeError("short response from tb_tpu_fpga")
            n = vals[pos]
            if n < 0:
                raise RuntimeError(f"TPU refused {OP_NAMES.get(line[0], line[0])} {list(line[1:4])}: "
                                   "arguments out of range")
            if line is parts:
                out = np.asarray(vals[pos + 1:pos + 1 + n], dtype=np.int64)
                if len(out) != n or (expect is not None and n != expect):
                    raise RuntimeError(
                        f"malformed FPGA response: declared {n}, expected {expect}, received {len(out)}")
            elif n != 0:
                raise RuntimeError(f"malformed FPGA response to a write: {n}")
            pos += 1 + n
        return out

    def _wr_words(self, sel, addr, words):
        """Queue writes; they go out with the next command, in the same transaction."""
        words = np.asarray(words, dtype=np.int64).reshape(-1)
        if addr < 0 or addr + words.size > MEM_WORDS.get(sel, 0) or sel == SEL_Y:
            raise ValueError(f"write of {words.size} words at {addr} doesn't fit memory {sel}")
        off = 0
        while off < words.size:
            chunk = words[off:off + WR_CHUNK]
            parts = [OP_WR, sel, addr + off, int(chunk.size)]
            parts.extend(int(v) for v in chunk)
            self._pending.append(parts)
            off += int(chunk.size)

    def cycles(self):
        """Clock cycles since reset: (with the core running, with the core or the host port working)."""
        busy, total = self._txn([OP_CYCLES], expect=2)
        return int(busy), int(total)

    def _store_weight(self, weight):
        """Weights larger than 1024 words stay in SRAM, keyed by their content, so the same tensor
        is uploaded once however the caller passes it. Smaller ones go to scratch every time."""
        w = np.ascontiguousarray(weight, dtype=np.float32)
        if w.size > WMEM_WORDS - SCRATCH:
            raise ValueError(f"weight of {w.size} words doesn't fit the {WMEM_WORDS - SCRATCH}-word scratch")
        key = None
        if w.size > 1024:
            key = (w.shape, hashlib.blake2b(w.tobytes(), digest_size=16).digest())
            hit = self._wmap.get(key)
            if hit is not None:
                return hit
        bits = to_f32(w.ravel())
        if key is not None and self._heap + w.size <= SCRATCH:
            addr = self._heap
            self._heap += int(w.size)
            self._wmap[key] = (addr, w.shape)
        else:
            addr = SCRATCH
        self._wr_words(SEL_W, addr, bits)
        return addr, w.shape

    def linear(self, x, weight, weight_scale=None, bias=None):
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        w = np.asarray(weight, dtype=np.float32)
        if w.ndim != 2 or x.size != w.shape[1]:
            raise ValueError("linear requires a 2-D weight and matching vector")
        m, k = w.shape
        if m > MEM_WORDS[SEL_Y] or k > MEM_WORDS[SEL_X]:
            raise ValueError(f"GEMV {m}x{k} exceeds the core's {MEM_WORDS[SEL_Y]} outputs or {MEM_WORDS[SEL_X]} inputs")
        addr, _ = self._store_weight(w)
        self._wr_words(SEL_X, 0, to_f32(x))
        tiles = (m + 3) // 4 * ((k + 63) // 64)
        self.gemm_tiles += tiles
        self.counts[CMD_GEMM_CLR] += 1
        self.counts[CMD_GEMM_ACC] += max(tiles - 1, 0)
        y = from_f32(self._txn([OP_GEMV, m, k, addr], expect=m, cycles=m * k))
        if bias is not None:
            y = self.vadd(y, bias)
        return y

    def _chunks(self, n):
        for i in range(0, n, VLEN):
            yield i, min(n, i + VLEN)

    def vadd(self, x, y):
        self.counts[CMD_VADD] += 1
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if x.size != y.size:
            raise ValueError("vector sizes must match")
        out = np.empty(x.size, dtype=np.float64)
        for a, b in self._chunks(x.size):
            self._wr_words(SEL_X, 0, to_f32(x[a:b]))
            self._wr_words(SEL_G, 0, to_f32(y[a:b]))
            out[a:b] = from_f32(self._txn([OP_VADD, b - a], expect=b - a))
        return out

    def vmul(self, x, scale):
        self.counts[CMD_VMUL] += 1
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        scale_q = int(to_f32(np.array([float(scale)]))[0])
        out = np.empty(x.size, dtype=np.float64)
        for a, b in self._chunks(x.size):
            self._wr_words(SEL_X, 0, to_f32(x[a:b]))
            out[a:b] = from_f32(self._txn([OP_VMUL, b - a, scale_q], expect=b - a))
        return out

    def vmul_elem(self, x, y):
        self.counts[CMD_VMUL] += 1
        shape = np.asarray(x).shape
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if x.size != y.size:
            raise ValueError("vector sizes must match")
        out = np.empty(x.size, dtype=np.float64)
        for a, b in self._chunks(x.size):
            self._wr_words(SEL_X, 0, to_f32(x[a:b]))
            self._wr_words(SEL_G, 0, to_f32(y[a:b]))
            out[a:b] = from_f32(self._txn([OP_VMUL_ELEM, b - a], expect=b - a))
        return out.reshape(shape)

    def rmsnorm(self, x, gamma, eps=1e-5):
        self.counts[CMD_RMSNORM] += 1
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        gamma = np.asarray(gamma, dtype=np.float64).reshape(-1)
        if not 0 < x.size <= VLEN or gamma.size != x.size or eps <= 0:
            raise ValueError(f"RMSNorm requires 1..{VLEN} lanes, matching gamma and positive epsilon")
        eps_q = int(to_f32(np.array([float(eps)]))[0])
        self._wr_words(SEL_X, 0, to_f32(x))
        self._wr_words(SEL_G, 0, to_f32(gamma))
        return from_f32(self._txn([OP_RMSNORM, x.size, eps_q], expect=x.size))

    def silu(self, x, use_lut=True):
        x = np.asarray(x, dtype=np.float64)
        shape = x.shape
        xf = x.reshape(-1)
        if use_lut:
            self.counts[CMD_ACT_LUT] += 1
        else:
            self.counts[CMD_SILU] += 1
        out = np.empty(xf.size, dtype=np.float64)
        for a, b in self._chunks(xf.size):
            self._wr_words(SEL_X, 0, to_f32(xf[a:b]))
            out[a:b] = from_f32(self._txn([OP_SILU, b - a, 0], expect=b - a))
        return out.reshape(shape)

    def softmax(self, logits, valid_len=None):
        self.counts[CMD_SOFTMAX] += 1
        z = np.asarray(logits, dtype=np.float64).reshape(-1)
        if z.size == 0:
            return np.zeros(0, dtype=np.float64)
        if z.size > VLEN or (valid_len is not None and not 0 <= valid_len <= z.size):
            raise ValueError(f"softmax requires <= {VLEN} lanes and valid_len within the vector")
        valid = 0 if valid_len is None else int(valid_len)
        self._wr_words(SEL_X, 0, to_f32(z))
        return from_f32(self._txn([OP_SOFTMAX, z.size, valid], expect=z.size))

    def rope(self, x, position, theta=10000.0):
        self.counts[CMD_ROPE] += 1
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        half = x.size // 2
        if x.size == 0 or x.size % 2 or x.size > self.vlen:
            raise ValueError("RoPE requires a positive even length <= VLEN")
        self._wr_words(SEL_X, 0, to_f32(x))
        self._wr_words(SEL_A, 0, rope_lut_angles(half, position, theta))
        return from_f32(self._txn([OP_ROPE, x.size], expect=x.size))
