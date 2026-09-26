"""Drive the clocked FPGA TPU (tpu_fpga.sv) through Icarus."""
from __future__ import annotations

import atexit
import os
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

OP_WR = 1
OP_RD = 2
OP_GEMV = 3
OP_VADD = 4
OP_VMUL = 5
OP_VMUL_ELEM = 6
OP_SILU = 7
OP_RMSNORM = 8
OP_SOFTMAX = 9
OP_ROPE = 10
OP_QUIT = 11

SEL_W, SEL_X, SEL_Y, SEL_G, SEL_A = range(5)
VLEN = 176
WMEM_WORDS = 524288
SCRATCH = WMEM_WORDS - 65536
WR_CHUNK = 2048

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
        self._wkeep = []
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
        print("compiling tpu_fpga.sv (1 MAC/cycle GEMV, VLEN=176, on-chip weights) ...", flush=True)
        proc = subprocess.run(
            [
                iverilog,
                *(["-B", str(self._lib)] if self._lib else []),
                "-g2012",
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
            if self._ready.exists():  # TB idle; otherwise just kill it
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

    def _write_cmd(self, parts):
        with open(self._cmd, "w", encoding="ascii", newline="\n") as f:
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
    def _wait_flag(self, path, timeout, what):
        t0 = time.monotonic()
        while not path.exists():
            if self._proc.poll() is not None:
                tail = ""
                try:
                    self._logf.flush()
                    tail = self._log.read_text(encoding="utf-8", errors="replace")[-2000:]
                except Exception:
                    pass
                raise RuntimeError(f"vvp exited while waiting for {what}\n" + tail)
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(f"TPU {what} timed out after {timeout}s")
            time.sleep(0.002)

    def _begin_txn(self):
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("vvp is not running — FPGA TPU sim died")
        self._wait_flag(self._ready, 30, "READY (previous command not acknowledged)")
        _unlink_flag(self._ready)

    def _end_txn(self, expect=None, timeout=600, quit=False):
        self._go.write_text("1", encoding="ascii")
        self._wait_flag(self._done, timeout, "command")
        raw = self._rsp.read_text(encoding="ascii").split()
        _unlink_flag(self._go)
        if not quit:
            self._wait_flag(self._ready, 30, "acknowledgement of GO removal")
        _unlink_flag(self._done)
        vals = [int(x) for x in raw]
        if not vals:
            raise RuntimeError("empty response from tb_tpu_fpga")
        n = vals[0]
        if n < 0 or len(vals) != n + 1 or (expect is not None and n != expect):
            raise RuntimeError(
                f"malformed FPGA response: declared {n}, expected {expect}, received {len(vals)-1}"
            )
        out = np.asarray(vals[1:1 + n], dtype=np.int64)
        if expect is not None and expect != 0:
            out = out[:expect]
        return out

    def _txn(self, parts, expect=None, timeout=600, quit=False):
        self._begin_txn()
        self._write_cmd(parts)
        return self._end_txn(expect=expect, timeout=timeout, quit=quit)

    def _wr_words(self, sel, addr, words):
        words = np.asarray(words, dtype=np.int64).reshape(-1)
        off = 0
        while off < words.size:
            chunk = words[off:off + WR_CHUNK]
            parts = [OP_WR, sel, addr + off, int(chunk.size)]
            parts.extend(int(v) for v in chunk)
            self._txn(parts, expect=0, timeout=30 + chunk.size * 0.002)
            off += int(chunk.size)

    def _store_weight(self, weight):
        key = id(weight)
        hit = self._wmap.get(key)
        if hit is not None:
            return hit
        w = np.ascontiguousarray(weight, dtype=np.float32)
        bits = to_f32(w.ravel())
        if w.size > 1024 and self._heap + w.size <= SCRATCH:
            addr = self._heap
            self._heap += int(w.size)
            self._wkeep.append(weight)
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
        addr, _ = self._store_weight(w)
        self._wr_words(SEL_X, 0, to_f32(x))
        tiles = (m + 3) // 4 * ((k + 63) // 64)
        self.gemm_tiles += tiles
        self.counts[CMD_GEMM_CLR] += 1
        self.counts[CMD_GEMM_ACC] += max(tiles - 1, 0)
        y = from_f32(self._txn([OP_GEMV, m, k, addr], expect=m, timeout=max(120, 30 + m * k * 0.002)))
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
