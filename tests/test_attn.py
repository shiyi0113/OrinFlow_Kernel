import torch
import torch.nn.functional as F
import ofk

WARMUP = 10
ITERS  = 100


def cuda_time_ms(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def ref_attn(q, k, v):
    """Numerically stable reference in float32."""
    scale = q.size(-1) ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale    # (B,H,M,N)
    scores = scores - scores.amax(dim=-1, keepdim=True)      # stability
    w = torch.softmax(scores, dim=-1)
    return torch.matmul(w, v)


def test_flash_attn():
    print("正确性验证 flash_attn...")
    cases = [
        (1,  1,  32,  32,  64),   # minimal
        (2,  8, 128, 128,  64),   # typical
        (1,  4, 256, 256, 128),
        (2, 16, 512, 512,  64),
    ]
    for B, H, M, N, d in cases:
        q = torch.randn(B, H, M, d, device="cuda")
        k = torch.randn(B, H, N, d, device="cuda")
        v = torch.randn(B, H, N, d, device="cuda")

        out = torch.ops.ofk.flash_attn(q, k, v)
        ref = ref_attn(q, k, v)
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
        print(f"  (B={B}, H={H:2d}, M={M:4d}, N={N:4d}, d={d}): PASS")
    print("flash_attn 正确性: PASS\n")


def bench_flash_attn():
    print("性能对比 flash_attn vs F.scaled_dot_product_attention")
    configs = [
        (1, 32,  512,  512, 128),
        (1, 32, 1024, 1024, 128),
        (2, 16, 2048, 2048,  64),
    ]
    print(f"{'B':>2} {'H':>3} {'M':>5} {'N':>5} {'d':>4}  │"
          f" {'flash_attn':>12}  │ {'torch SDPA':>12}  │ {'speedup':>8}")
    print("─" * 70)
    for B, H, M, N, d in configs:
        q = torch.randn(B, H, M, d, device="cuda")
        k = torch.randn(B, H, N, d, device="cuda")
        v = torch.randn(B, H, N, d, device="cuda")

        ms_ours = cuda_time_ms(lambda: torch.ops.ofk.flash_attn(q, k, v))
        ms_sdpa = cuda_time_ms(lambda: F.scaled_dot_product_attention(q, k, v))
        print(f"{B:>2} {H:>3} {M:>5} {N:>5} {d:>4}  │"
              f" {ms_ours:>10.3f}ms  │ {ms_sdpa:>10.3f}ms  │ {ms_sdpa/ms_ours:>7.2f}x")


if __name__ == "__main__":
    test_flash_attn()
    bench_flash_attn()
