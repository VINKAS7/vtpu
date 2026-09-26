// Memory and vector sizes for the Stories260K clocked TPU.
// Keep in sync with VLEN / WMEM_WORDS in tpu_fpga.py.

`ifndef HW_PARAMS_SVH
`define HW_PARAMS_SVH

`define VLEN            176
`define XMEM_WORDS      256
`define YMEM_WORDS      512
`define WMEM_WORDS      524288

`endif
