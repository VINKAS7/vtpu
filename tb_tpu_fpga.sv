`timescale 1ns/1ps
`include "hw_params.svh"

// File-IPC host for tpu_fpga. Weights stay in on-chip SRAM after LOAD.

module tb_tpu_fpga;
    localparam SEL_Y = 3'd2;
    localparam OP_WR = 1, OP_RD = 2, OP_GEMV = 3, OP_VADD = 4, OP_VMUL = 5;
    localparam OP_VMUL_ELEM = 6, OP_SILU = 7, OP_RMSNORM = 8, OP_SOFTMAX = 9, OP_ROPE = 10, OP_QUIT = 11;

    reg clk, rst, start, host_en, host_we;
    reg [3:0] opcode;
    reg [15:0] arg_m, arg_k, arg_len;
    reg [31:0] arg_addr, arg_aux, host_addr, host_wdata;
    reg [2:0] host_sel;
    wire [31:0] host_rdata;
    wire busy, done;

    tpu_fpga DUT (
        .clk(clk), .rst(rst), .start(start), .opcode(opcode),
        .arg_m(arg_m), .arg_k(arg_k), .arg_addr(arg_addr),
        .arg_len(arg_len), .arg_aux(arg_aux),
        .host_en(host_en), .host_we(host_we), .host_sel(host_sel),
        .host_addr(host_addr), .host_wdata(host_wdata), .host_rdata(host_rdata),
        .busy(busy), .done(done)
    );

    always #5 clk = ~clk;

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
                integer guard;
                guard = 0;
                while (done == 1'b0) begin
                    @(posedge clk);
                    guard = guard + 1;
                    if (guard > 2000000) begin
                        $display("tb_tpu_fpga: watchdog timeout op=%0d m=%0d k=%0d", op, m_i, k_i);
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
            case (opcode_i)
                OP_WR: begin
                    got = $fscanf(cmd_fd, "%d %d %d", sel, addr, n);
                    for (i = 0; i < n; i = i + 1) begin
                        got = $fscanf(cmd_fd, "%d", tmp);
                        host_write(sel[2:0], addr + i, tmp);
                    end
                    $fwrite(rsp_fd, "0\n");
                end
                OP_RD: begin
                    got = $fscanf(cmd_fd, "%d %d %d", sel, addr, n);
                    $fwrite(rsp_fd, "%0d\n", n);
                    for (i = 0; i < n; i = i + 1) begin
                        host_read(sel[2:0], addr + i, tmp);
                        $fwrite(rsp_fd, "%0d\n", $signed(tmp));
                    end
                end
                OP_GEMV: begin
                    got = $fscanf(cmd_fd, "%d %d %d", m, k, addr);
                    run_op(4'd1, m[15:0], k[15:0], 16'd0, addr, 32'b0);
                    write_y_rsp(m);
                end
                OP_VADD: begin
                    got = $fscanf(cmd_fd, "%d", n);
                    run_op(4'd2, 16'd0, 16'd0, n[15:0], 32'b0, 32'b0);
                    write_y_rsp(n);
                end
                OP_VMUL: begin
                    got = $fscanf(cmd_fd, "%d %d", n, aux);
                    run_op(4'd3, 16'd0, 16'd0, n[15:0], 32'b0, aux);
                    write_y_rsp(n);
                end
                OP_VMUL_ELEM: begin
                    got = $fscanf(cmd_fd, "%d", n);
                    run_op(4'd4, 16'd0, 16'd0, n[15:0], 32'b0, 32'b0);
                    write_y_rsp(n);
                end
                OP_SILU: begin
                    got = $fscanf(cmd_fd, "%d %d", n, aux);
                    run_op(4'd5, 16'd0, 16'd0, n[15:0], 32'b0, aux);
                    write_y_rsp(n);
                end
                OP_RMSNORM: begin
                    got = $fscanf(cmd_fd, "%d %d", n, aux);
                    run_op(4'd6, 16'd0, 16'd0, n[15:0], 32'b0, aux);
                    write_y_rsp(n);
                end
                OP_SOFTMAX: begin
                    got = $fscanf(cmd_fd, "%d %d", n, aux);
                    run_op(4'd7, 16'd0, 16'd0, n[15:0], 32'b0, aux);
                    write_y_rsp(n);
                end
                OP_ROPE: begin
                    got = $fscanf(cmd_fd, "%d", n);
                    run_op(4'd8, 16'd0, 16'd0, n[15:0], 32'b0, 32'b0);
                    write_y_rsp(n);
                end
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
                    $fwrite(rsp_fd, "0\n");
                end
            endcase
            finish_txn();
        end
    end
endmodule
