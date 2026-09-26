# vTPU: Stories260K on a clocked TPU core

This package runs the Stories260K Llama model (F32 GGUF, included) on a clocked SystemVerilog TPU core, simulated cycle by cycle in Icarus Verilog. Python schedules the model and sends each GEMV or vector operation to the core. A pure-NumPy backend with the same API serves as the reference.

## Setup and run

You need Python 3.10+ and Icarus Verilog (`iverilog`, `vvp`) on your PATH. On Windows, pass `--tool-dir <iverilog\bin>` instead.

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python run_stories260k_tpu.py --prompt "Once upon a time" --max-new-tokens 24               # clocked core (default)
python run_stories260k_tpu.py --backend numpy --prompt "Once upon a time" --max-new-tokens 24
python verify.py --tokens 8      # full regression suite, about 1.5 min
python test_ipc_race.py          # handshake robustness (macOS/Linux)
```

Generation uses greedy decoding and is deterministic. The core generates about 5 s per token on an M-series Mac, and its tokens match the NumPy backend exactly.

## Files

| File | Role |
|---|---|
| `tpu_fpga.sv` | Clocked TPU core, plus the `tpu_fp32_pkg` FP32 arithmetic |
| `hw_params.svh` | Memory and vector sizes |
| `tb_tpu_fpga.sv` | Testbench that turns file commands into host-port writes and `start`/`done` handshakes |
| `tpu_fpga.py` | Python host: compiles and launches `vvp`, uploads weights once, issues commands |
| `tpu_llm_sim.py` | NumPy reference TPU (same API) and FP32 bit helpers |
| `stories260k_tpu.py` | Llama forward pass (attention with KV cache, SwiGLU FFN) expressed as TPU commands |
| `run_stories260k_tpu.py` | GGUF loader, SentencePiece BPE tokenizer, CLI |
| `verify.py` | Regression suite: FP32 bit-exactness, kernels, and full model against NumPy and a dense reference |
| `test_ipc_race.py` | Pauses `vvp` after every command, to check that the host↔simulator handshake cannot deadlock |
| `ISSUES.md` | Known bugs and improvements, most critical first |

## The core (`tpu_fpga.sv`)

| Block | What it is |
|---|---|
| GEMV | 1 FP32 multiply-accumulate per cycle (`m·k` cycles per GEMV) over weights held in on-chip SRAM |
| Vector ops | add, scale, elementwise multiply, SiLU, RMSNorm, softmax (with valid-length mask), RoPE; 1 element per cycle, `VLEN` = 176 |
| Weight SRAM | 524288 × 32b (about 2 MB). The whole model (about 1 MB) is uploaded once; the top 64K words are scratch |
| Activations | `x` 256 words, `y` 512 words, `g` 176 words, RoPE angles 88 words |

The host keeps the KV cache and uploads the current K/V slice to scratch for each attention GEMV.

## Limits

This is **simulation**, not hardware:

- The FP32 add/mul/div/exp/rsqrt/sin/cos functions use Verilog `real`. To synthesize the core, replace them with pipelined FP IP, register the memory reads, and add multi-cycle latency to the FSM. See `ISSUES.md`.
- No FPGA timing, area, power, or throughput numbers are claimed.

Only the implemented Llama configuration is supported: F32 tensors, full-head adjacent-pair RoPE, RMSNorm, SiLU gating, grouped-query attention, SentencePiece score-based BPE, and a 128-token context. Requests beyond the context window are rejected.
