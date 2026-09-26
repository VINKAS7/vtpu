# vTPU: Stories260K on a clocked TPU core

This package runs the Stories260K Llama model (F32 GGUF, included) on a clocked SystemVerilog TPU core, simulated cycle by cycle in Icarus Verilog. Python schedules the model and sends each GEMV or vector operation to the core. A pure-NumPy backend with the same API serves as the reference.

## Setup and run

You need Python 3.10+ and Icarus Verilog (`iverilog`, `vvp`) on your PATH. On Windows, pass `--tool-dir <iverilog\bin>` instead.

With [direnv](https://direnv.net), run `direnv allow` once: entering the folder then creates `.venv` with `requirements.txt` if it's missing, and activates it. Without direnv:

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python run_stories260k_tpu.py --prompt "Once upon a time" --max-new-tokens 24               # clocked core (default)
python run_stories260k_tpu.py --backend numpy --prompt "Once upon a time" --max-new-tokens 24
python verify.py --tokens 8      # full regression suite, about 1 min
python test_ipc_race.py          # handshake robustness and hang reports (macOS/Linux)
```

Generation uses greedy decoding and is deterministic, and its tokens match the NumPy backend exactly.

## Speed

`run_stories260k_tpu.py` prints the speed at the end. For 24 tokens from "Once upon a time", on an Apple M4 Pro:

| | |
|---|---:|
| tokens per second on this machine (the simulation's wall clock) | about 0.2 (5 s per token) |
| clock cycles per token, with the core running | 315,916 |
| clock cycles per token, with the core or the host port working | 360,762 |
| of those cycles, simulated per second on this machine | about 69,000 |

The cycle counts are exact and repeat from run to run. The testbench's waits for the host's files aren't counted.

The testbench's clock has a 10 ns period, a nominal 100 MHz, but only in simulated time: the core isn't synthesized yet, so it has no real clock (see "Limits"). At 100 MHz, 315,916 cycles would be about 317 tokens/s. That's a what-if for the RTL as written, not a measurement.

## Files

| File | Role |
|---|---|
| `tpu_fpga.sv` | Clocked TPU core, plus the `tpu_fp32_pkg` FP32 arithmetic |
| `hw_params.svh` | Memory sizes, opcodes and memory selects: the one table the RTL, the testbench and `tpu_fpga.py` all read |
| `tb_tpu_fpga.sv` | Testbench that turns file commands into host-port writes and `start`/`done` handshakes, and counts clock cycles |
| `tpu_fpga.py` | Python host: compiles and launches `vvp`, uploads weights once, issues commands |
| `tpu_llm_sim.py` | NumPy reference TPU (same API) and FP32 bit helpers |
| `stories260k_tpu.py` | Llama forward pass (attention with KV cache, SwiGLU FFN) expressed as TPU commands |
| `run_stories260k_tpu.py` | GGUF loader, SentencePiece BPE tokenizer, CLI |
| `verify.py` | Regression suite: FP32 bit-exactness, kernels, and full model against NumPy and a dense reference |
| `test_ipc_race.py` | Pauses `vvp` after every command, to check that the host↔simulator handshake cannot deadlock, and stops it, to check that a hang is reported within seconds |

## The core (`tpu_fpga.sv`)

| Block | What it is |
|---|---|
| GEMV | 1 FP32 multiply-accumulate per cycle (`m·k` cycles per GEMV) over weights held in on-chip SRAM |
| Vector ops | add, scale, elementwise multiply, SiLU, RMSNorm, softmax (with valid-length mask), RoPE; 1 element per cycle, `VLEN` = 176 |
| Weight SRAM | 524288 × 32b (about 2 MB). The whole model (about 1 MB) is uploaded once; the top 64K words are scratch |
| Activations | `x` 256 words, `y` 512 words, `g` 176 words, RoPE angles 88 words |

The host keeps the KV cache and uploads the current K/V slice to scratch for each attention GEMV.

**Host protocol.** Python and the testbench talk through files, with a strict 4-phase handshake (`go`, `done`, `ready`). One transaction carries the queued writes and the next command together, and the testbench answers each one in turn. A command whose arguments don't fit the memories is refused: the core raises `error`, the testbench answers `-1`, and Python raises an error naming the command. Python checks the same bounds before sending. A transaction that doesn't finish in time (20 s, plus an allowance per cycle and per word) stops the run, with the last command, the flag files and the end of the `vvp` log, and kills `vvp`.

## Limits

This is **simulation**, not hardware:

- The FP32 add/mul/div/exp/rsqrt/sin/cos functions use Verilog `real`. To synthesize the core, replace them with pipelined FP IP, register the memory reads, and add multi-cycle latency to the FSM.
- `weight_mem` is read combinationally, with a multiplier in the address path and up to 3 reads per cycle. BRAM or URAM needs a registered read. At about 2 MB, it also needs URAM on F2, or DDR with a cache.
- `OP_ROPE` writes two `y_mem` words per cycle, and the host and the core share `x_mem`, `y_mem` and `g_mem` with no arbitration: the RAMs need two ports, or the FSM needs more cycles.
- RMSNorm computes `1/len` with a `real` division; hardware needs a constant or a reciprocal table.
- No FPGA timing, area, power, or throughput numbers are claimed.

Next steps for speed, in order of payoff: a pipe or VPI transport instead of files; attention on chip, with the KV cache in on-chip memory; and a parallel GEMV.

Only the implemented Llama configuration is supported: F32 tensors, full-head adjacent-pair RoPE, RMSNorm, SiLU gating, grouped-query attention, SentencePiece score-based BPE, and a 128-token context. Requests beyond the context window are rejected.
