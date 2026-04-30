import torch
import ofk
import time

def benchmark(fn, warmup=20, repeat=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat * 1000  # ms

def bench(n):
    x = torch.randn(n, dtype=torch.float32, device="cuda")

    t_ofk   = benchmark(lambda: torch.ops.ofk.reduce(x))
    t_torch = benchmark(lambda: x.sum())

    bw_peak = 448.0  # GB/s，按你的 GPU 改
    bytes_read = n * 4 / 1e9  # GB
    bw_ofk   = bytes_read / (t_ofk   / 1000)
    bw_torch = bytes_read / (t_torch / 1000)

    print(f"n={n:>10,}  |  ofk {t_ofk:.3f} ms ({bw_ofk:.1f} GB/s)  |  "
          f"torch {t_torch:.3f} ms ({bw_torch:.1f} GB/s)  |  "
          f"slowdown {t_ofk/t_torch:.1f}x  |  "
          f"util {bw_ofk/bw_peak*100:.1f}%")

if __name__ == "__main__":
    print(f"{'n':>12}  |  {'ofk':^28}  |  {'torch':^28}  |  slowdown  |  util")
    print("-" * 100)
    for n in [1 << k for k in range(10, 26)]:  # 1K ~ 32M
        bench(n)
