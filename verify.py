"""Regression suite for the clocked TPU (tpu_fpga.sv); requires NumPy + Icarus."""
from pathlib import Path
import argparse
import subprocess
import sys
import tempfile
import time

import numpy as np

from run_stories260k_tpu import GgufTokenizer, load_gguf_llama, parse_gguf
from stories260k_tpu import Stories260KTPU
from tpu_fpga import VLEN, TpuLlmFpga
from tpu_llm_sim import TpuLlm

ROOT = Path(__file__).resolve().parent


def reference(ids, cfg, w):
    """Independent dense causal Llama reference, no TPU methods or KV cache."""
    x = w['wte'][ids].astype(np.float64)
    def rms(x, g):
        return x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + cfg.rms_eps) * g
    def rot(a):
        angle = np.arange(len(ids))[:,None,None] * cfg.rope_theta**(-np.arange(d//2)/float(d//2))[None,None,:]
        z = np.empty_like(a)
        z[...,0::2] = a[...,0::2]*np.cos(angle)-a[...,1::2]*np.sin(angle)
        z[...,1::2] = a[...,0::2]*np.sin(angle)+a[...,1::2]*np.cos(angle)
        return z
    d = cfg.n_embd // cfg.n_head
    for i in range(cfg.n_layer):
        p = f'h.{i}.'
        a = rms(x, w[p+'attn_norm.w'])
        q = rot((a @ w[p+'q.w'].T).reshape(-1,cfg.n_head,d))
        k = rot((a @ w[p+'k.w'].T).reshape(-1,cfg.n_kv_head,d))
        v = (a @ w[p+'v.w'].T).reshape(-1,cfg.n_kv_head,d)
        k = np.repeat(k, cfg.n_head//cfg.n_kv_head, axis=1)
        v = np.repeat(v, cfg.n_head//cfg.n_kv_head, axis=1)
        z = np.einsum('thd,shd->hts',q,k)/np.sqrt(d)
        z[:,np.triu_indices(len(ids),1)[0],np.triu_indices(len(ids),1)[1]] = -np.inf
        z = np.exp(z-z.max(axis=-1,keepdims=True)); z /= z.sum(axis=-1,keepdims=True)
        x += np.einsum('hts,shd->thd',z,v).reshape(-1,cfg.n_embd) @ w[p+'o.w'].T
        a = rms(x,w[p+'ffn_norm.w'])
        g = a @ w[p+'gate.w'].T
        x += (g/(1+np.exp(-g)) * (a @ w[p+'up.w'].T)) @ w[p+'down.w'].T
    return rms(x[-1],w['norm_w']) @ w['lm_head'].T


def reject(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('invalid input accepted')


def fp32_pkg_tests(t):
    """Bit-exact checks of tpu_fp32_pkg (the RTL's FP32 arithmetic) against NumPy."""
    rng = np.random.default_rng(42)
    vals = rng.integers(0, 2**32, 400, dtype=np.uint32).view(np.float32)
    vals = vals[np.isfinite(vals)]
    lines = []
    for a, b in zip(vals[::2], vals[1::2]):
        ab = int(np.asarray(a).view(np.uint32)); bb = int(np.asarray(b).view(np.uint32))
        for op, fn in [('add', np.add), ('sub', np.subtract), ('mul', np.multiply), ('div', np.divide)]:
            with np.errstate(all='ignore'):
                eb = int(np.asarray(fn(a, b), dtype=np.float32).view(np.uint32))
            lines.append(f'if(fp32_{op}(32\'h{ab:08x},32\'h{bb:08x}) !== 32\'h{eb:08x}) $fatal(1,"random {op} {ab:08x} {bb:08x}");')
    body = '\n        '.join(lines)
    src = f'''
module fp32_test;
    import tpu_fp32_pkg::*;
    initial begin
        if(real_to_fp32(1.0+2.0**-24) !== 32'h3f800000) $fatal(1,"tie-even down");
        if(real_to_fp32(1.0+3.0*2.0**-24) !== 32'h3f800002) $fatal(1,"tie-even up");
        if(real_to_fp32(2.0**-149) !== 32'h00000001) $fatal(1,"subnormal");
        if(fp32_mul(32'h00800000,32'h3f000000) !== 32'h00400000) $fatal(1,"underflow");
        if(fp32_div(32'hbf800000,0) !== 32'hff800000) $fatal(1,"div sign");
        if(fp32_div(0,0) !== 32'h7fc00000) $fatal(1,"div nan");
        {body}
        $display("PASS: FP32 package rounding, subnormals, exceptions, {len(lines)} random ops bit-exact");
        $finish;
    end
endmodule
'''
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp); (p/'fp32_test.sv').write_text(src)
        iverilog = str(Path(t.vvp_bin).with_name('iverilog' + Path(t.vvp_bin).suffix))
        subprocess.run([iverilog, *(['-B',str(t._lib)] if t._lib else []), '-g2012', '-I', str(ROOT),
                        '-s', 'fp32_test', '-o', str(p/'t.vvp'), str(ROOT/'tpu_fpga.sv'), str(p/'fp32_test.sv')],
                       check=True, capture_output=True, text=True)
        r = subprocess.run([t.vvp_bin, *(['-M','-','-M',str(t._lib)] if t._lib else []), str(p/'t.vvp')],
                           capture_output=True, text=True, timeout=60)
        print(r.stdout.strip(), flush=True)
        r.check_returncode()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tool-dir'); ap.add_argument('--tokens', type=int, default=8)
    args = ap.parse_args(); rng = np.random.default_rng(20260926); start = time.monotonic()
    def check(name, a, b, rtol=2e-5, atol=2e-6):
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
        print('PASS:', name, flush=True)

    # Host-side model, tokenizer and input validation (NumPy backend).
    cfg, w, tok = load_gguf_llama(ROOT/'stories260K.gguf')
    cpu = Stories260KTPU(cfg, w); ids = tok.encode('Once upon a time')
    assert ids == [1,403,407,261,378]
    for text in ['héllo 世界🙂', ' leading space', '', 'Once upon a time']:
        assert tok.decode(tok.encode(text)) == text
    custom = GgufTokenizer(['a','b','c','ab','bc'], scores=[0,0,0,1,2], add_bos=False, add_space_prefix=False)
    assert custom.encode('abc') == [0,4]
    print('PASS: tokenizer score-order merges and UTF-8 round trip', flush=True)
    logits, _ = cpu.forward(ids)
    check('independent dense reference', logits, reference(ids, cfg, w), atol=3e-5)
    last = None
    for token in ids:
        inc, last = cpu.forward([token], last)
    check('cached vs full prefill', inc, logits)
    reject(lambda: cpu.forward([])); reject(lambda: cpu.forward([-1])); reject(lambda: cpu.forward([cfg.vocab_size]))
    reject(lambda: cpu.forward([1]*(cfg.block_size+1)))
    reject(lambda: cpu.generate([1], -1)); reject(lambda: cpu.generate([1]*128, 2))
    cache = [(np.zeros((128,4,8)), np.zeros((128,4,8))) for _ in range(5)]
    reject(lambda: cpu.forward([1], cache))
    cache[0] = (np.zeros((1,4,8)), np.zeros((1,4,8)))
    reject(lambda: cpu.forward([1], cache))
    data = (ROOT/'stories260K.gguf').read_bytes()
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp)/'bad.gguf'
        f.write_bytes(data[:-4]); reject(lambda: parse_gguf(f))
        bad = bytearray(data); bad[4:8] = (1).to_bytes(4, 'little'); f.write_bytes(bad)
        reject(lambda: parse_gguf(f))
    print('PASS: invalid model/token/cache/context/GGUF inputs rejected', flush=True)

    # Clocked TPU kernels and full model against the NumPy reference.
    with TpuLlmFpga(tool_dir=args.tool_dir) as hw:
        fp32_pkg_tests(hw)
        ref = TpuLlm()
        for n in [1, 3, 4, 63, 64, 65, 128, 172, VLEN]:
            x = rng.normal(size=n); y = rng.normal(size=n)
            check(f'vadd/{n}', hw.vadd(x,y), ref.vadd(x,y))
            check(f'vmul/{n}', hw.vmul(x,.125), ref.vmul(x,.125))
            check(f'elementwise/{n}', hw.vmul_elem(x,y), ref.vmul_elem(x,y))
            check(f'rmsnorm/{n}', hw.rmsnorm(x,y), ref.rmsnorm(x,y))
            check(f'softmax/{n}', hw.softmax(x), ref.softmax(x))
            check(f'masked-softmax/{n}', hw.softmax(x,1), ref.softmax(x,1))
        for m, k in [(1,1), (4,4), (7,9), (64,64), (172,64), (64,172), (512,64)]:
            x = rng.normal(size=k); a = rng.normal(size=(m,k))
            check(f'gemv/{m}x{k}', hw.linear(x,a), ref.linear(x,a), atol=2e-5)
        x = np.array([-100,-60,-20,-15,-10,-1,0,1,10,15,20,60,100], dtype=np.float32)
        check('silu', hw.silu(x), ref.silu(x))
        for n in [2, 8, 64]:
            for pos in [0, 1, 31, 127]:
                x = rng.normal(size=n); check(f'rope/{n}/{pos}', hw.rope(x,pos), ref.rope(x,pos))
        reject(lambda: hw.softmax(np.zeros(VLEN+1)))
        reject(lambda: hw.rmsnorm(np.zeros(VLEN+1), np.zeros(VLEN+1)))
        reject(lambda: hw.rope(np.zeros(9), 1))
        reject(lambda: hw.vadd([1], [1,2]))
        reject(lambda: hw.linear([1], np.ones((2,2))))
        print('PASS: host rejects out-of-range vector and GEMV shapes', flush=True)

        model = Stories260KTPU(cfg, w, tpu=hw)
        hc = nc = None; generated = []; maxerr = 0.0
        for step in range(args.tokens):
            inp = ids if step == 0 else [generated[-1]]
            a, hc = model.forward(inp, hc); b, nc = cpu.forward(inp, nc)
            check(f'full-model logits/token-{step+1}', a, b, atol=6e-5)
            assert int(np.argmax(a)) == int(np.argmax(b))
            generated.append(int(np.argmax(a)))
            maxerr = max(maxerr, float(np.max(np.abs(a-b))))
            print('TPU TOKEN:', generated[-1], repr(tok.decode(ids+generated)), flush=True)
        print('tpu cmds:', hw.summary(), flush=True)
    print(f'ALL CHECKS PASSED in {time.monotonic()-start:.0f}s; max logit error {maxerr:.2e}:',
          repr(tok.decode(ids+generated)), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
