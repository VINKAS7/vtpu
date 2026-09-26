`timescale 1ns/1ps
`include "hw_params.svh"

// Clocked TPU core for Stories260K.
// - GEMV: one FP32 multiply-accumulate per cycle over on-chip weights
// - On-chip weight / activation SRAM; weights uploaded once by the host
// - Vector units: add, scale, elementwise mul, SiLU, RMSNorm, softmax, RoPE
// FP32 add/mul/exp/rsqrt still use simulation reals; swap for vendor IP on FPGA.

package tpu_fp32_pkg;
    function automatic real fp32_to_real;
        input [31:0] f;
        reg [10:0] de;
        real v;
        begin
            if (f[30:23] == 0) begin
                v = $itor({1'b0, f[22:0]}) * (2.0 ** -149);
                fp32_to_real = f[31] ? -v : v;
            end else if (f[30:23] == 255)
                fp32_to_real = $bitstoreal({f[31], 11'h7FF, f[22:0], 29'b0});
            else begin
                de = {3'b0, f[30:23]} + 11'd896;
                fp32_to_real = $bitstoreal({f[31], de, f[22:0], 29'b0});
            end
        end
    endfunction

    function automatic [31:0] real_to_fp32;
        input real r;
        reg [63:0] d, sig, q, remnant, halfway;
        integer e, shift;
        begin
            d = $realtobits(r);
            e = d[62:52] - 1023;
            if (d[62:52] == 2047)
                real_to_fp32 = {d[63], 8'hFF, (d[51:0] == 0 ? 23'b0 : 23'h400000)};
            else if (d[62:52] == 0 || e < -150)
                real_to_fp32 = {d[63], 31'b0};
            else if (e > 127)
                real_to_fp32 = {d[63], 8'hFF, 23'b0};
            else begin
                sig = {11'b0, 1'b1, d[51:0]};
                shift = e < -126 ? -e - 97 : 29;
                q = sig >> shift;
                remnant = sig & ((64'd1 << shift) - 1);
                halfway = 64'd1 << (shift - 1);
                if (remnant > halfway || (remnant == halfway && q[0])) q = q + 1;
                if (e < -126)
                    real_to_fp32 = {d[63], 7'b0, q[23:0]};
                else begin
                    if (q[24]) begin q = q >> 1; e = e + 1; end
                    e = e + 127;
                    if (e >= 255) real_to_fp32 = {d[63], 8'hFF, 23'b0};
                    else real_to_fp32 = {d[63], e[7:0], q[22:0]};
                end
            end
        end
    endfunction

    function automatic is_nan;
        input [31:0] a;
        begin is_nan = a[30:23] == 8'hff && a[22:0] != 0; end
    endfunction

    function automatic [31:0] fp32_add;
        input [31:0] a, b;
        begin fp32_add = real_to_fp32(fp32_to_real(a) + fp32_to_real(b)); end
    endfunction

    function automatic [31:0] fp32_sub;
        input [31:0] a, b;
        begin fp32_sub = real_to_fp32(fp32_to_real(a) - fp32_to_real(b)); end
    endfunction

    function automatic [31:0] fp32_mul;
        input [31:0] a, b;
        begin fp32_mul = real_to_fp32(fp32_to_real(a) * fp32_to_real(b)); end
    endfunction

    function automatic [31:0] fp32_div;
        input [31:0] a, b;
        begin
            if ((a[30:23] == 255 && a[22:0] != 0) ||
                (b[30:23] == 255 && b[22:0] != 0) ||
                (a[30:0] == 0 && b[30:0] == 0) ||
                (a[30:0] == 31'h7f800000 && b[30:0] == 31'h7f800000))
                fp32_div = 32'h7fc00000;
            else if (b[30:0] == 0) fp32_div = {a[31]^b[31], 8'hff, 23'b0};
            else fp32_div = real_to_fp32(fp32_to_real(a) / fp32_to_real(b));
        end
    endfunction

    function automatic [31:0] fp32_rsqrt;
        input [31:0] a;
        begin
            if (a[30:0] == 0) fp32_rsqrt = {a[31], 8'hff, 23'b0};
            else if (a[31] || (a[30:23] == 255 && a[22:0] != 0)) fp32_rsqrt = 32'h7fc00000;
            else fp32_rsqrt = real_to_fp32(1.0 / $sqrt(fp32_to_real(a)));
        end
    endfunction

    function automatic [31:0] fp32_exp;
        input [31:0] a;
        real x;
        begin
            x = fp32_to_real(a);
            if (x > 89.0) fp32_exp = 32'h7f800000;
            else if (x < -104.0) fp32_exp = 0;
            else fp32_exp = real_to_fp32($exp(x));
        end
    endfunction

    function automatic [31:0] fp32_silu;
        input [31:0] a;
        real x, ex;
        begin
            x = fp32_to_real(a);
            if (x >= 0.0) fp32_silu = real_to_fp32(x / (1.0 + $exp(-x)));
            else if (a == 32'hff800000) fp32_silu = 32'h80000000;
            else begin
                ex = $exp(x);
                fp32_silu = real_to_fp32(x * ex / (1.0 + ex));
            end
        end
    endfunction

endpackage


module tpu_fpga (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire [3:0]  opcode,
    input  wire [15:0] arg_m,
    input  wire [15:0] arg_k,
    input  wire [31:0] arg_addr,
    input  wire [15:0] arg_len,
    input  wire [31:0] arg_aux,
    input  wire        host_en,
    input  wire        host_we,
    input  wire [2:0]  host_sel,
    input  wire [31:0] host_addr,
    input  wire [31:0] host_wdata,
    output reg  [31:0] host_rdata,
    output reg         busy,
    output reg         done,
    output reg         error     // with done: the command's arguments were out of range; nothing ran
);
    import tpu_fp32_pkg::*;

    localparam OP_GEMV      = `OP_GEMV;
    localparam OP_VADD      = `OP_VADD;
    localparam OP_VMUL      = `OP_VMUL;
    localparam OP_VMUL_ELEM = `OP_VMUL_ELEM;
    localparam OP_SILU      = `OP_SILU;
    localparam OP_RMSNORM   = `OP_RMSNORM;
    localparam OP_SOFTMAX   = `OP_SOFTMAX;
    localparam OP_ROPE      = `OP_ROPE;

    localparam SEL_W = `SEL_W, SEL_X = `SEL_X, SEL_Y = `SEL_Y, SEL_G = `SEL_G, SEL_A = `SEL_A;

    localparam S_IDLE   = 4'd0;
    localparam S_FETCH  = 4'd1;
    localparam S_MUL    = 4'd2;
    localparam S_RED    = 4'd3;
    localparam S_ACC    = 4'd4;
    localparam S_VEC    = 4'd5;
    localparam S_REDUCE = 4'd6;
    localparam S_SCALE  = 4'd7;
    localparam S_NORM   = 4'd8;
    localparam S_DONE   = 4'd9;

    localparam integer VLEN  = `VLEN;

    (* ram_style = "block" *) reg [31:0] weight_mem [0:`WMEM_WORDS-1];
    (* ram_style = "block" *) reg [31:0] x_mem      [0:`XMEM_WORDS-1];
    (* ram_style = "block" *) reg [31:0] y_mem      [0:`YMEM_WORDS-1];
    (* ram_style = "block" *) reg [31:0] g_mem      [0:VLEN-1];
    (* ram_style = "block" *) reg [31:0] a_mem      [0:VLEN/2-1];

    reg [3:0]  state;
    reg [3:0]  op;
    reg [15:0] m_reg, k_reg, len_reg, idx, row0, k0, work_len;
    reg [31:0] wbase, aux_reg, acc;
    reg [31:0] sumsq, maxv, inv, scale;

    // A command's arguments must fit the memories it touches, or it would read or write out of
    // range. The host checks them too; this is the last line of defence.
    wire [47:0] gemv_end = {16'b0, arg_addr} + arg_m * arg_k;
    wire gemv_bad = arg_m > `YMEM_WORDS || arg_k > `XMEM_WORDS || gemv_end > `WMEM_WORDS;
    wire vec_bad  = arg_len > VLEN || (opcode == OP_ROPE && arg_len[0]);
    wire op_known = opcode >= OP_GEMV && opcode <= OP_ROPE;

    always @(posedge clk) begin
        if (host_en && host_we) begin
            case (host_sel)
                SEL_W:  if (host_addr < `WMEM_WORDS) weight_mem[host_addr] <= host_wdata;
                SEL_X:  if (host_addr < `XMEM_WORDS) x_mem[host_addr]      <= host_wdata;
                SEL_G:  if (host_addr < VLEN)        g_mem[host_addr]      <= host_wdata;
                SEL_A:  if (host_addr < VLEN/2)      a_mem[host_addr]      <= host_wdata;
                default: ;
            endcase
        end
        if (host_en && !host_we) begin
            case (host_sel)
                SEL_W:  host_rdata <= (host_addr < `WMEM_WORDS) ? weight_mem[host_addr] : 32'b0;
                SEL_X:  host_rdata <= (host_addr < `XMEM_WORDS) ? x_mem[host_addr]      : 32'b0;
                SEL_Y:  host_rdata <= (host_addr < `YMEM_WORDS) ? y_mem[host_addr]      : 32'b0;
                SEL_G:  host_rdata <= (host_addr < VLEN)        ? g_mem[host_addr]      : 32'b0;
                SEL_A:  host_rdata <= (host_addr < VLEN/2)      ? a_mem[host_addr]      : 32'b0;
                default: host_rdata <= 32'b0;
            endcase
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            busy  <= 1'b0;
            done  <= 1'b0;
            error <= 1'b0;
            idx   <= 0;
            row0  <= 0;
            k0    <= 0;
            acc   <= 0;
        end else begin
            done <= 1'b0;
            case (state)
                S_IDLE: if (start) begin
                    op      <= opcode;
                    m_reg   <= arg_m;
                    k_reg   <= arg_k;
                    wbase   <= arg_addr;
                    len_reg <= arg_len;
                    aux_reg <= arg_aux;
                    busy    <= 1'b1;
                    idx     <= 0;
                    row0    <= 0;
                    k0      <= 0;
                    acc     <= 0;
                    sumsq   <= 0;
                    error   <= !op_known || (opcode == OP_GEMV ? gemv_bad : vec_bad);
                    if (!op_known || (opcode == OP_GEMV ? gemv_bad : vec_bad))
                        state <= S_DONE;
                    else if (arg_len == 0 && opcode != OP_GEMV)
                        state <= S_DONE;
                    else if (opcode == OP_GEMV)
                        state <= (arg_m == 0 || arg_k == 0) ? S_DONE : S_FETCH;
                    else if (opcode == OP_RMSNORM || opcode == OP_SOFTMAX)
                        state <= S_REDUCE;
                    else
                        state <= S_VEC;
                    if (opcode == OP_SOFTMAX) begin
                        if (arg_aux == 0 || arg_aux[15:0] > arg_len)
                            work_len <= arg_len;
                        else
                            work_len <= arg_aux[15:0];
                    end
                end

                // One MAC per cycle: row0 walks the outputs, k0 the inner dimension.
                S_FETCH: begin
                    if (k0 == 0)
                        acc <= fp32_mul(weight_mem[wbase + {16'b0, row0} * {16'b0, k_reg} + {16'b0, k0}],
                                        x_mem[k0]);
                    else
                        acc <= fp32_add(acc,
                                        fp32_mul(weight_mem[wbase + {16'b0, row0} * {16'b0, k_reg} + {16'b0, k0}],
                                                 x_mem[k0]));
                    if (k0 + 16'd1 == k_reg) begin
                        y_mem[row0] <= (k0 == 0)
                            ? fp32_mul(weight_mem[wbase + {16'b0, row0} * {16'b0, k_reg}], x_mem[0])
                            : fp32_add(acc,
                                       fp32_mul(weight_mem[wbase + {16'b0, row0} * {16'b0, k_reg} + {16'b0, k0}],
                                                x_mem[k0]));
                        k0 <= 0;
                        if (row0 + 16'd1 == m_reg)
                            state <= S_DONE;
                        else
                            row0 <= row0 + 16'd1;
                    end else
                        k0 <= k0 + 16'd1;
                end

                S_VEC: begin
                    if (idx >= len_reg)
                        state <= S_DONE;
                    else begin
                        case (op)
                            OP_VADD:
                                y_mem[idx] <= fp32_add(x_mem[idx], g_mem[idx]);
                            OP_VMUL:
                                y_mem[idx] <= fp32_mul(x_mem[idx], aux_reg);
                            OP_VMUL_ELEM:
                                y_mem[idx] <= fp32_mul(x_mem[idx], g_mem[idx]);
                            OP_SILU:
                                y_mem[idx] <= fp32_silu(x_mem[idx]);
                            OP_ROPE: begin
                                y_mem[idx]     <= fp32_sub(
                                    fp32_mul(x_mem[idx],     real_to_fp32($cos(fp32_to_real(a_mem[idx[15:1]])))),
                                    fp32_mul(x_mem[idx + 1], real_to_fp32($sin(fp32_to_real(a_mem[idx[15:1]])))));
                                y_mem[idx + 1] <= fp32_add(
                                    fp32_mul(x_mem[idx],     real_to_fp32($sin(fp32_to_real(a_mem[idx[15:1]])))),
                                    fp32_mul(x_mem[idx + 1], real_to_fp32($cos(fp32_to_real(a_mem[idx[15:1]])))));
                            end
                            default: ;
                        endcase
                        if (op == OP_ROPE)
                            idx <= idx + 16'd2;
                        else
                            idx <= idx + 16'd1;
                    end
                end

                S_REDUCE: begin
                    if (op == OP_RMSNORM) begin
                        if (idx < len_reg) begin
                            sumsq <= fp32_add(sumsq, fp32_mul(x_mem[idx], x_mem[idx]));
                            idx   <= idx + 16'd1;
                        end else begin
                            scale <= fp32_rsqrt(fp32_add(fp32_mul(sumsq, real_to_fp32(1.0 / $itor(len_reg))), aux_reg));
                            idx   <= 0;
                            state <= S_SCALE;
                        end
                    end else begin
                        if (idx == 0) begin
                            maxv <= x_mem[0];
                            idx  <= 16'd1;
                        end else if (idx < work_len) begin
                            // A NaN wins and stays, so it reaches every output, as in NumPy.
                            if (!is_nan(maxv) && (is_nan(x_mem[idx]) ||
                                                  fp32_to_real(x_mem[idx]) > fp32_to_real(maxv)))
                                maxv <= x_mem[idx];
                            idx <= idx + 16'd1;
                        end else begin
                            idx   <= 0;
                            sumsq <= 0;
                            state <= S_SCALE;
                        end
                    end
                end

                S_SCALE: begin
                    if (op == OP_RMSNORM) begin
                        if (idx < len_reg) begin
                            y_mem[idx] <= fp32_mul(fp32_mul(x_mem[idx], scale), g_mem[idx]);
                            idx <= idx + 16'd1;
                        end else
                            state <= S_DONE;
                    end else begin
                        if (idx < work_len) begin
                            y_mem[idx] <= fp32_exp(fp32_sub(x_mem[idx], maxv));
                            sumsq <= fp32_add(sumsq, fp32_exp(fp32_sub(x_mem[idx], maxv)));
                            idx   <= idx + 16'd1;
                        end else begin
                            inv   <= fp32_div(32'h3f800000, sumsq);  // the max lane adds exp(0) = 1, so 0 can't occur
                            idx   <= 0;
                            state <= S_NORM;
                        end
                    end
                end

                S_NORM: begin
                    if (idx < len_reg) begin
                        if (idx < work_len)
                            y_mem[idx] <= fp32_mul(y_mem[idx], inv);
                        else
                            y_mem[idx] <= 32'b0;
                        idx <= idx + 16'd1;
                    end else
                        state <= S_DONE;
                end

                S_DONE: begin
                    busy  <= 1'b0;
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
