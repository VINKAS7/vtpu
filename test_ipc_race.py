"""Regression: the host/TB file handshake must survive vvp being descheduled.

After every command, vvp is SIGSTOPped right after it creates DONE and resumed
50 ms later. The old host recreated GO during that pause; the TB never saw it
disappear, and both sides waited forever. POSIX only (needs SIGSTOP).
"""
from __future__ import annotations

import os
import signal
import sys
import threading

import numpy as np

import time

import tpu_fpga
from tpu_fpga import SEL_X, TpuLlmFpga
from tpu_llm_sim import TpuLlm, from_f32


def main():
    if not hasattr(signal, "SIGSTOP"):
        print("SKIP: needs SIGSTOP")
        return
    ref = TpuLlm()
    rng = np.random.default_rng(0)
    with TpuLlmFpga() as tpu:
        wait_flag = tpu._wait_flag

        def resume(pid):
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass

        def preempting_wait(path, timeout, what):
            wait_flag(path, timeout, what)
            if path == tpu._done:
                os.kill(tpu._proc.pid, signal.SIGSTOP)
                threading.Timer(0.05, resume, (tpu._proc.pid,)).start()

        tpu._wait_flag = preempting_wait
        for i in range(40):
            # Back-to-back writes (expect=0) are what used to deadlock.
            x = rng.standard_normal(64).astype(np.float32)
            y = rng.standard_normal(64).astype(np.float32)
            w = rng.standard_normal((8, 64)).astype(np.float32)
            tpu._wr_words(SEL_X, 0, [0])
            np.testing.assert_allclose(tpu.vadd(x, y), ref.vadd(x, y), rtol=1e-6)
            np.testing.assert_allclose(tpu.linear(x, w), ref.linear(x, w), rtol=1e-5, atol=1e-5)
        got = from_f32(tpu._txn([2, SEL_X, 0, 1], expect=1))  # OP_RD one word
        assert got.shape == (1,)
        tpu._wait_flag = wait_flag
    print("PASS: 40 preempted rounds (200+ transactions), no deadlock")

    # A simulator that stops answering is reported within seconds, with what the host was doing,
    # and killed.
    tpu_fpga.TXN_BASE_S = 3.0
    with TpuLlmFpga() as tpu:
        pid = tpu._proc.pid
        os.kill(pid, signal.SIGSTOP)
        t0 = time.monotonic()
        try:
            tpu.vadd([1.0], [2.0])
        except TimeoutError as e:
            took = time.monotonic() - t0
            assert "last command: VADD" in str(e) and "flag files:" in str(e), e
            assert took < 10, took
            assert tpu._proc is None, "vvp should be killed"
        else:
            raise AssertionError("a stopped simulator went unnoticed")
    print(f"PASS: a hung simulator is reported after {took:.0f}s, with its state, and killed")


if __name__ == "__main__":
    sys.exit(main())
