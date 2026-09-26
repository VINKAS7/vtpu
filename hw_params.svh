// Memory sizes, opcodes and memory selects for the Stories260K clocked TPU: the one table that
// tpu_fpga.sv, tb_tpu_fpga.sv and tpu_fpga.py (which parses this file) all use.

`ifndef HW_PARAMS_SVH
`define HW_PARAMS_SVH

`define VLEN            176
`define XMEM_WORDS      256
`define YMEM_WORDS      512
`define WMEM_WORDS      524288
`define SCRATCH_WORDS   65536

// Commands. The core runs 3 to 10; the testbench handles WR, RD and QUIT itself.
`define OP_WR           1
`define OP_RD           2
`define OP_GEMV         3
`define OP_VADD         4
`define OP_VMUL         5
`define OP_VMUL_ELEM    6
`define OP_SILU         7
`define OP_RMSNORM      8
`define OP_SOFTMAX      9
`define OP_ROPE         10
`define OP_QUIT         11
`define OP_CYCLES       12   // testbench: cycles so far with the core running, and with the core or host port working

// Memory selects for host reads and writes.
`define SEL_W           0
`define SEL_X           1
`define SEL_Y           2
`define SEL_G           3
`define SEL_A           4

`endif
