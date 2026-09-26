`timescale 1ns/1ps
`include "hw_params.svh"

// File-IPC host for tpu_fpga. Weights stay in on-chip SRAM after LOAD.
// One transaction's command file holds one or more commands, one per line, run in order. Each
// command answers with a count and that many values, or with -1 if it was refused (arguments out
// of range, or an unknown opcode).

module tb_tpu_fpga;
    localparam SEL_Y = `SEL_Y;
    localparam OP_WR = `OP_WR, OP_RD = `OP_RD, OP_GEMV = `OP_GEMV, OP_QUIT = `OP_QUIT;

    reg clk, rst, start, host_en, host_we;
    reg [3:0] opcode;
    reg [15:0] arg_m, arg_k, arg_len;
    reg [31:0] arg_addr, arg_aux, host_addr, host_wdata;
    reg [2:0] host_sel;
    wire [31:0] host_rdata;
    wire busy, done, error;

    tpu_fpga DUT (
        .clk(clk), .rst(rst), .start(start), .opcode(opcode),
        .arg_m(arg_m), .arg_k(arg_k), .arg_addr(arg_addr),
        .arg_len(arg_len), .arg_aux(arg_aux),
        .host_en(host_en), .host_we(host_we), .host_sel(host_sel),
        .host_addr(host_addr), .host_wdata(host_wdata), .host_rdata(host_rdata),
        .busy(busy), .done(done), .error(error)
    );

    always #5 clk = ~clk;  // a 10 ns period: a nominal 100 MHz, in simulated time only

    // Clock cycles since reset: those with the core running a command, and those with the core or
    // the host port working. The rest is the testbench waiting for the host's files.
    reg [63:0] busy_cycles, all_cycles;
    always @(posedge clk)
        if (rst) begin
            busy_cycles <= 0;
            all_cycles  <= 0;
        end else begin
            if (busy) busy_cycles <= busy_cycles + 1;
            if (busy || host_en) all_cycles <= all_cycles + 1;
        end

    reg [8*512-1:0] p_ready, p_go, p_done, p_cmd, p_rsp;

    function integer file_exists;
        input [8*512-1:0] name;
        integer fd;
        begin
            fd = $fopen(name, "r");
            if (fd != 0) begin
                $fclose(fd);
                file_exists = 1;
            end else
                file_exists = 0;
        end
    endfunction

    task wait_flag;
        input [8*512-1:0] name;
        input present;
        begin
            while (file_exists(name) != present)
                #1000;
            repeat (2) @(posedge clk);
        end
    endtask

    integer cmd_fd, rsp_fd, done_fd, got, opcode_i, n, m, k, addr, aux, i, tmp, sel;

    // The words each host-visible memory holds; an access past the end is refused.
    function integer mem_words;
        input integer sel_i;
        begin
            case (sel_i)
                `SEL_W: mem_words = `WMEM_WORDS;
                `SEL_X: mem_words = `XMEM_WORDS;
                `SEL_Y: mem_words = `YMEM_WORDS;
                `SEL_G: mem_words = `VLEN;
                `SEL_A: mem_words = `VLEN / 2;
                default: mem_words = 0;
            endcase
        end
    endfunction

    task host_write;
        input [2:0] sel_i;
        input [31:0] addr_i;
        input [31:0] data_i;
        begin
            @(negedge clk);
            host_en = 1; host_we = 1; host_sel = sel_i;
            host_addr = addr_i; host_wdata = data_i;
            @(negedge clk);
            host_en = 0; host_we = 0;
        end
    endtask

    task host_read;
        input [2:0] sel_i;
        input [31:0] addr_i;
        output [31:0] data_o;
        begin
            @(negedge clk);
            host_en = 1; host_we = 0; host_sel = sel_i; host_addr = addr_i;
            @(posedge clk);
            @(negedge clk);
            data_o = host_rdata;
            host_en = 0;
        end
    endtask

    task run_op;
        input [3:0] op;
        input [15:0] m_i, k_i, len_i;
        input [31:0] addr_i, aux_i;
        begin
            @(negedge clk);
            opcode = op; arg_m = m_i; arg_k = k_i; arg_len = len_i;
            arg_addr = addr_i; arg_aux = aux_i; start = 1;
            @(negedge clk);
            start = 0;
            begin : WAIT_DONE
                // Every command takes at most one cycle per MAC or a few per lane; allow 4x that.
                integer guard, limit;
                guard = 0;
                limit = 4 * (m_i * k_i + 4 * len_i) + 1000;
                while (done == 1'b0) begin
                    @(posedge clk);
                    guard = guard + 1;
                    if (guard > limit) begin
                        $display("tb_tpu_fpga: watchdog timeout op=%0d m=%0d k=%0d len=%0d after %0d cycles",
                                 op, m_i, k_i, len_i, guard);
                        $finish;
                    end
                end
            end
            @(negedge clk);
        end
    endtask

    task write_y_rsp;
        input integer count;
        begin
            if (error) begin
                $fwrite(rsp_fd, "-1\n");
                disable write_y_rsp;
            end
            $fwrite(rsp_fd, "%0d\n", count);
            for (i = 0; i < count; i = i + 1) begin
                host_read(SEL_Y, i, tmp);
                $fwrite(rsp_fd, "%0d\n", $signed(tmp));
            end
        end
    endtask

    task finish_txn;
        begin
            $fclose(cmd_fd);
            $fclose(rsp_fd);
            done_fd = $fopen(p_done, "w");
            $fclose(done_fd);
            wait_flag(p_go, 0);
            done_fd = $fopen(p_ready, "w");
            $fclose(done_fd);
        end
    endtask

    initial begin
        clk = 0; rst = 1; start = 0; host_en = 0; host_we = 0;
        opcode = 0; arg_m = 0; arg_k = 0; arg_len = 0; arg_addr = 0; arg_aux = 0;
        host_sel = 0; host_addr = 0; host_wdata = 0;
        if (!$value$plusargs("READY=%s", p_ready)) p_ready = "ready";
        if (!$value$plusargs("GO=%s", p_go))       p_go = "go";
        if (!$value$plusargs("DONE=%s", p_done))   p_done = "done";
        if (!$value$plusargs("CMD=%s", p_cmd))     p_cmd = "cmd.txt";
        if (!$value$plusargs("RSP=%s", p_rsp))     p_rsp = "rsp.txt";

        repeat (8) @(posedge clk);
        rst = 0;
        repeat (4) @(posedge clk);
        $display("tb_tpu_fpga: ready VLEN=%0d WMEM=%0d", `VLEN, `WMEM_WORDS);
        done_fd = $fopen(p_ready, "w");
        if (done_fd == 0) begin
            $display("tb_tpu_fpga: cannot write READY");
            $finish;
        end
        $fclose(done_fd);

        forever begin
            wait_flag(p_go, 1);
            cmd_fd = $fopen(p_cmd, "r");
            rsp_fd = $fopen(p_rsp, "w");
            if (cmd_fd == 0 || rsp_fd == 0) begin
                $display("tb_tpu_fpga: cannot open cmd/rsp");
                $finish;
            end
            got = $fscanf(cmd_fd, "%d", opcode_i);
            while (got == 1) begin
                case (opcode_i)
                    OP_WR: begin
                        got = $fscanf(cmd_fd, "%d %d %d", sel, addr, n);
                        if (addr < 0 || n < 0 || addr + n > mem_words(sel) || sel == `SEL_Y) begin
                            for (i = 0; i < n; i = i + 1)
                                got = $fscanf(cmd_fd, "%d", tmp);
                            $fwrite(rsp_fd, "-1\n");
                        end else begin
                            for (i = 0; i < n; i = i + 1) begin
                                got = $fscanf(cmd_fd, "%d", tmp);
                                host_write(sel[2:0], addr + i, tmp);
                            end
                            $fwrite(rsp_fd, "0\n");
                        end
                    end
                    OP_RD: begin
                        got = $fscanf(cmd_fd, "%d %d %d", sel, addr, n);
                        if (addr < 0 || n < 0 || addr + n > mem_words(sel))
                            $fwrite(rsp_fd, "-1\n");
                        else begin
                            $fwrite(rsp_fd, "%0d\n", n);
                            for (i = 0; i < n; i = i + 1) begin
                                host_read(sel[2:0], addr + i, tmp);
                                $fwrite(rsp_fd, "%0d\n", $signed(tmp));
                            end
                        end
                    end
                    OP_GEMV: begin
                        got = $fscanf(cmd_fd, "%d %d %d", m, k, addr);
                        run_op(OP_GEMV, m[15:0], k[15:0], 16'd0, addr, 32'b0);
                        write_y_rsp(m);
                    end
                    `OP_VADD, `OP_VMUL_ELEM, `OP_ROPE: begin
                        got = $fscanf(cmd_fd, "%d", n);
                        run_op(opcode_i[3:0], 16'd0, 16'd0, n[15:0], 32'b0, 32'b0);
                        write_y_rsp(n);
                    end
                    `OP_VMUL, `OP_SILU, `OP_RMSNORM, `OP_SOFTMAX: begin
                        got = $fscanf(cmd_fd, "%d %d", n, aux);
                        run_op(opcode_i[3:0], 16'd0, 16'd0, n[15:0], 32'b0, aux);
                        write_y_rsp(n);
                    end
                    `OP_CYCLES:
                        $fwrite(rsp_fd, "2\n%0d\n%0d\n", busy_cycles, all_cycles);
                    OP_QUIT: begin
                        $fwrite(rsp_fd, "0\n");
                        $fclose(cmd_fd);
                        $fclose(rsp_fd);
                        done_fd = $fopen(p_done, "w");
                        $fclose(done_fd);
                        $display("tb_tpu_fpga: quit");
                        $finish;
                    end
                    default: begin
                        $display("tb_tpu_fpga: bad opcode %0d", opcode_i);
                        $fwrite(rsp_fd, "-1\n");
                    end
                endcase
                got = $fscanf(cmd_fd, "%d", opcode_i);
            end
            finish_txn();
        end
    end
endmodule
