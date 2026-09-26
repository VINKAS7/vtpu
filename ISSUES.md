# vTPU: bugs and improvements (most critical first)

Scope: the clocked TPU path, from top to bottom:
`run_stories260k_tpu.py` → `stories260k_tpu.py` → `tpu_fpga.py` (host) → file IPC → `tb_tpu_fpga.sv` → `tpu_fpga.sv`.

| # | Issue | Status |
|---|---|---|
| 1 | File-handshake deadlock ("stuck at token N") | **Fixed** |
| 2 | Hangs surface only after long timeouts | Open |
| 3 | Default backend had no tests | **Fixed** (`verify.py` now targets `tpu_fpga.sv`) |
| 4 | README claimed a 256-MAC array and on-chip KV | **Fixed** (docs corrected, dead `kv_mem` removed) |
| 5 | RTL not synthesizable | Open |
| 6 | Memory bounds not checked | Open |
| 7 | Performance | Open |
| 8 | Numerical edge cases | Open |
| 9 | Consistency and hygiene | Partly fixed |

The "stuck" bug was **not** in the datapath. It was a race in the Python↔Icarus handshake, and it depended on timing, which is why it showed up at a "random" token such as the 14th. I reproduced it deterministically (item 1).

---

## 1. ~~CRITICAL: the file handshake can deadlock (this is the "stuck at token N" bug)~~ FIXED

> **Status:** fixed in `tpu_fpga.py`.
> - The host now runs a strict 4-phase handshake. It starts a transaction only when `ready` exists, and every command except QUIT waits for the testbench's `ready` before returning.
> - `close()` sends QUIT only when the testbench is idle.
> - Regression test: `test_ipc_race.py`. It pauses `vvp` with SIGSTOP for 50 ms after every command, over 200+ transactions.
> - Cost: about 30% slower, because each write now waits for its acknowledgement. The pipe/VPI transport below removes that cost.

**Where:** `tpu_fpga.py:_begin_txn` / `_end_txn` and `tb_tpu_fpga.sv:finish_txn`.

**How the protocol runs:**

| step | Python | Testbench |
|---|---|---|
| 1 | writes `cmd.txt`, creates `go` | sees `go`, runs the command, writes `rsp.txt`, creates `done` |
| 2 | sees `done`, reads rsp, **deletes `go`** | **polls until `go` is gone** (`wait_flag(p_go, 0)`) |
| 3 | waits for `ready` **only if `expect != 0`** | creates `ready`, then waits for `go` again |

For write transactions (`expect=0`, which are the majority: every `_wr_words`), Python skips step 3. It starts the next transaction immediately and creates `go` again. If `vvp` isn't scheduled during the few milliseconds between Python deleting `go` and recreating it, the testbench never sees `go` absent:

- the testbench waits forever for `go` to disappear;
- Python waits forever for `done`, which it already deleted.

**Repro:** send `SIGSTOP` to `vvp` right after it creates `done`, let Python run its normal expect=0 path plus the next transaction, then send `SIGCONT`. Result: `DEADLOCK: vvp still waiting for GO to disappear; Python waiting for DONE`, every time. On a loaded machine, Windows with Defender or indexing, or a slow disk, the OS causes this pause by itself. The more transactions a token issues, the more likely it gets; there are about 1–2k per token.

**Second, related bug:** `_begin_txn` deletes `ready` without waiting for it. A late `ready` from the previous transaction then survives into the next one. The next `expect != 0` transaction sees that stale `ready`, skips the acknowledgement, and opens the same window again.

**Fix (minimal):**
- Always wait for `ready` after deleting `go`, including when `expect == 0`.
- In `_begin_txn`, *wait for* `ready`; don't delete it blindly.
- Better: put a monotonically increasing transaction id in `go` and `done`, so neither side can confuse the state of one transaction with another.

**Fix (proper):** replace file polling with a pipe or socket (e.g. `$fgets` on a named FIFO), or with cocotb/VPI. This also removes most of the per-token overhead (item 5).

## 2. HIGH: a hang looks like "stuck" for up to 10 minutes, then fails without a clear message

- Timeouts in `_txn`: vector ops default to **600 s**, GEMV is at least 120 s, and writes are 30 s. After a deadlock, the user watches a frozen terminal for up to 10 minutes before seeing `TimeoutError`.
- The error doesn't say which command or which token it was on, or what the flag files looked like (`go`/`done`/`ready`).
- `close()` then sends `OP_QUIT` into the deadlocked simulator and waits another 5 s.
- **Fix:**
  - Add a heartbeat and use short timeouts that scale with the command's cycle count.
  - On timeout, dump the state of the flag files, the last command, and the tail of the `vvp` log.
  - Kill `vvp` directly when a transaction timed out.

## 3. ~~HIGH: the default backend has no tests~~ FIXED

> **Status:** the legacy 4×4 backend (`tpu_llm_full.sv`, `tb_tpu_host.sv`, `tpu_rtl.py`) was removed, together with its reports and logs. `verify.py` now tests `tpu_fpga.sv`:
> - `tpu_fp32_pkg` is bit-exact against NumPy (800 random ops, plus rounding, subnormal and exception cases);
> - every kernel is compared with NumPy, up to VLEN and GEMV 512×64;
> - out-of-range shapes are rejected;
> - 8 full-model tokens match.
>
> `test_ipc_race.py` covers the handshake.

Still missing: a full 128-token-context run, and running both scripts in CI.

## 4. ~~HIGH: the README architecture doesn't match the RTL~~ FIXED (docs)

> **Status:** the README, header comments and log messages now describe the real core: 1 MAC per cycle, KV cache on the host. The unused `kv_mem`, the GELU path and the unused `hw_params.svh` defines were removed. Actually building the parallel GEMV and on-chip KV is still open (item 7).

| README says | RTL actually does |
|---|---|
| "4 output rows × 64 lanes, 256 MACs, every column used" | `S_FETCH` does **1 multiply-add per cycle**, one row and one k at a time (`tpu_fpga.sv:257–280`). `GEMV_ROWS`/`GEMV_LANES` are never used. |
| "On-chip KV SRAM" | `kv_mem` is declared and the host can write it, but the core never reads it. The KV cache lives in Python. |
| "Weights are uploaded once" | True for the linear layers. For attention, however, every head and every token re-uploads the K and V slices (`linear(q, kt)` and `linear(probs, V.T)`) to `SCRATCH`, so that traffic grows as O(t) per head per layer. |
| Control is "fetch → 64-wide multiply → add tree → accumulate" | There is no add tree. The accumulator is serial, so the result is also rounded in a different order than a tree would round it. |

Any throughput or cycle number derived from this core does not describe the claimed 256-MAC design.

## 5. MEDIUM: the RTL is not synthesizable as written, despite the `ram_style="block"` hints

- The FP32 functions are all `real`-based (`fp32_add`, `mul`, `div`, `rsqrt`, `exp`, `silu`, `$sin`, `$cos`). The README admits this. These must become pipelined FP IP, and the FSM then needs multi-cycle latencies. Today every operation is combinational within one cycle.
- `weight_mem` is read **combinationally**, with a multiplier in the address path (`wbase + row0*k_reg + k0`) and up to 3 identical reads in one cycle. That will not infer BRAM or URAM, which need a registered read. It also won't meet 100 MHz timing.
- `OP_ROPE` writes `y_mem[idx]` and `y_mem[idx+1]` and reads `x_mem[idx]` and `x_mem[idx+1]` in the same cycle, which needs 2 write ports on `y_mem`.
- `y_mem`, `x_mem` and `g_mem` are accessed from two `always` blocks: host and core. That works on true dual-port RAM, but nothing arbitrates, so the host can write while `busy`.
- About 2 MB of `weight_mem` (524288 × 32b) is far more than BRAM on most parts. It needs URAM on F2, or DDR/HBM with a cache.

## 6. MEDIUM: memory bounds are not checked, so errors are silent

- The RTL silently **drops** host writes beyond the end of a memory and reads back `0`. Nothing validates the core's GEMV arguments:
  - `k > XMEM_WORDS (256)` reads `x_mem` out of range and returns X;
  - `m > YMEM_WORDS (512)` writes out of range.
- `vocab = 512` is exactly `YMEM_WORDS`. Any model with a larger vocabulary silently corrupts the logits.
- `tpu_fpga.py:_store_weight`: when the heap is full, it falls back to `SCRATCH` **without checking that the tensor fits in the remaining 65536 words**. A larger tensor would get truncated without any error.
- `_store_weight` caches by `id(weight)`. It stays correct only because `_wkeep` pins every cached object. Callers that pass a float64 array get a fresh float32 copy on each call. Each copy is larger than 1024 words, so it gets a new heap slot every time, and the heap leaks until the scratch fallback above triggers.
- **Fix:**
  - Add `assert`/`ValueError` checks in `linear()` for `m ≤ 512`, `k ≤ 256` and `size ≤ free`.
  - Add an error or status bit in the RTL.
  - Key the weight cache by tensor name, not `id()`.

## 7. MEDIUM: performance (about 5 s per token for a 260K-parameter model)

- About 1–2k file transactions per token. Each one costs an `fsync`, Python's 2 ms polling sleeps, and the testbench's polling of the flag file every `#1000`.
- Every word goes through a 2-cycle `host_write`/`host_read` in the testbench, and `x` is rewritten before every GEMV.
- **Fix, in order of payoff:**
  1. Switch to a pipe or VPI transport (item 1).
  2. Batch commands into a queue or microcode, so that one transaction runs a whole layer.
  3. Add on-chip KV memory and do attention on chip.
  4. Actually parallelize the GEMV (item 4).

## 8. LOW: numerical and edge-case handling

- Softmax sets `inv = 1.0` when `sumsq == 0`, which hides a degenerate or NaN input. The max comparison uses `real >`, so a NaN never wins and gets silently skipped.
- RMSNorm computes `1/len` with `$itor` and `real` division. That's fine in simulation but needs a constant or reciprocal table in hardware.
- `fp32_to_real` for subnormals uses `$itor({1'b0, f[22:0]})`. This is correct but slow; fine for simulation only.
- The testbench watchdog is 2,000,000 cycles. That is fine for this model, but serial GEMV at 1 MAC per cycle would trip it on any model with 4× larger matrices.

## 9. LOW: consistency and hygiene

- Opcodes differ between layers: the testbench's `OP_GEMV = 3` becomes the core's `4'd1`, `OP_VADD = 4` becomes `4'd2`, and so on. This is easy to get wrong; share one table.
- `tpu_fpga.py` hardcodes `VLEN = 176` and `WMEM_WORDS = 524288`, which duplicates `hw_params.svh`. If they drift apart, nothing reports an error. Generate one from the other.
- ~~`hw_params.svh` defines symbols the RTL never uses~~. Fixed.
- ~~The README contains Windows-only paths~~. Fixed. `_ICARUS_HINTS` in `tpu_fpga.py` still only looks for Windows `.exe` installs; on macOS and Linux, the PATH is used.
- The folder is not a git repo.
