# benchmark_batch_all.py

import time
import numpy as np
import cupy as cp

import cpu_implementation_batched as cpu
import gpu_implementation_batched as gpu

def bench_batch(name, cpu_fn, gpu_fn, a_cpu, b_cpu, n_cpu, sync_gpu=True):
    a_gpu = cp.asarray(a_cpu)
    b_gpu = cp.asarray(b_cpu)
    n_gpu = cp.asarray(n_cpu, dtype=a_gpu.dtype)

    # warm up
    _ = gpu_fn(a_gpu, b_gpu, n_gpu)
    if sync_gpu:
        cp.cuda.Stream.null.synchronize()

    # CPU
    t0 = time.perf_counter()
    out_cpu = cpu_fn(a_cpu, b_cpu, n_cpu)
    t1 = time.perf_counter()
    t_cpu = t1 - t0

    # GPU
    t0 = time.perf_counter()
    out_gpu = gpu_fn(a_gpu, b_gpu, n_gpu)
    if sync_gpu:
        cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()
    t_gpu = t1 - t0

    # verify
    np.testing.assert_array_equal(out_cpu, cp.asnumpy(out_gpu))
    print(f"{name:35s} | CPU {t_cpu:8.4f}s   GPU {t_gpu:8.4f}s   speedup {t_cpu/t_gpu:6.2f}×")


def main():
    # 61-bit prime for Level-1
    N1 = 2305843009213693951

    # ----------------------------------------------------------------
    # Level-1: vectorized, huge batch
    # ----------------------------------------------------------------
    BATCH1 = 2_000_000
    rng = np.random.default_rng(42)
    a1 = rng.integers(0, N1,    size=BATCH1, dtype=np.int64)
    b1 = rng.integers(0, N1,    size=BATCH1, dtype=np.int64)
    e1 = rng.integers(0, 4096,  size=BATCH1, dtype=np.int64)

    print("\n=== Level-1 (vectorized) on batch =", BATCH1)
    bench_batch("naive_modular_multiplication",
                cpu.naive_modular_multiplication,
                gpu.naive_modular_multiplication,
                a1, b1, N1)

    bench_batch("naive_modular_exponentiation",
                cpu.naive_modular_exponentiation,
                gpu.naive_modular_exponentiation,
                a1, e1, N1)

    # ----------------------------------------------------------------
    # Level-2: loop-based on moderate batch, small bases/exponents
    # ----------------------------------------------------------------
    BATCH2   = 20_000
    MUL_IT   = 16       # naive add-loop repeats
    EXP_IT   = 8        # naive mul-loop repeats
    BASE_MUL = 1 << 32  # a < 2^32 => a*result ≤ 2^48
    BASE_EXP = 1 << 16  # a < 2^16 => base^8 ≤ 2^128 (fits in 128 but not 64!)
                        # However since we reduce mod N1 each mul, values remain < N1.

    print(f"\n=== Level-2 (loop-based) batch={BATCH2}, mul-it={MUL_IT}, exp-it={EXP_IT}")
    # for multiplication loop
    a2_mul = rng.integers(0, BASE_MUL, size=BATCH2, dtype=np.int64)
    b2_mul = np.full(BATCH2, MUL_IT, dtype=np.int64)
    bench_batch("naive_modular_multiplication_loop",
                cpu.naive_modular_multiplication_loop,
                gpu.naive_modular_multiplication_loop,
                a2_mul, b2_mul, N1)

    # # for exponentiation loop
    # a2_exp = rng.integers(0, BASE_EXP, size=BATCH2, dtype=np.int64)
    # b2_exp = np.full(BATCH2, EXP_IT, dtype=np.int64)
    # bench_batch("naive_modular_exponentiation_loop",
    #             cpu.naive_modular_exponentiation_loop,
    #             gpu.naive_modular_exponentiation_loop,
    #             a2_exp, b2_exp, N1)

    # # square‐and-multiply on same small batch
    # bench_batch("square_and_multiply_modular_exponentiation",
    #             cpu.square_and_multiply_modular_exponentiation,
    #             gpu.square_and_multiply_modular_exponentiation,
    #             a2_exp, b2_exp, N1)

    # ----------------------------------------------------------------
    # Level-3: Montgomery on a tiny prime + small batch
    # ----------------------------------------------------------------
    # choose a small prime < 2^31 to avoid any 64-bit overflow
    N3 = 2_147_483_647  
    BATCH3 = 2_000_000

    rng2 = np.random.default_rng(99)
    a3 = rng2.integers(0, N3, size=BATCH3, dtype=np.int64)
    b3 = rng2.integers(0, N3, size=BATCH3, dtype=np.int64)
    r3, nd3 = cpu.montgomery_precomputation(N3)

    print(f"\n=== Level-3 Montgomery on small prime {N3}, batch = {BATCH3}")

    cpu_mont = lambda x,y,n: cpu.montgomery_modular_multiplication(x,y,n,r3,nd3)
    gpu_mont = lambda x,y,n: gpu.montgomery_modular_multiplication(x,y,n,r3,nd3)
    bench_batch("montgomery_modular_multiplication",
                cpu_mont, gpu_mont,
                a3, b3, N3)

    cpu_opt = lambda x,y,n: cpu.optimized_montgomery_modular_multiplication(x,y,n,r3,nd3)
    gpu_opt = lambda x,y,n: gpu.optimized_montgomery_modular_multiplication(x,y,n,r3,nd3)
    bench_batch("optimized_montgomery_modular_multiplication",
                cpu_opt, gpu_opt,
                a3, b3, N3)


if __name__ == "__main__":
    main()