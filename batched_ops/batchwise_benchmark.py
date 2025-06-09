import time
import numpy as np
import cupy as cp
import os

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


def run_all_benchmarks(batch_size=2_000_000):
    """
    Run all benchmarks for a given batch size for Level 1, 2, 3, 4.
    Returns a dict: {method: (cpu_time, gpu_time)} for the main methods.
    """
    import time
    import numpy as np
    import cupy as cp
    import cpu_implementation_batched as cpu
    import gpu_implementation_batched as gpu

    results = {}

    # 61-bit prime for Level-1
    N1 = 2305843009213693951
    BATCH1 = batch_size
    rng = np.random.default_rng(42)
    a1 = rng.integers(0, N1,    size=BATCH1, dtype=np.int64)
    b1 = rng.integers(0, N1,    size=BATCH1, dtype=np.int64)
    e1 = rng.integers(0, 4096,  size=BATCH1, dtype=np.int64)

    def bench_batch_times(name, cpu_fn, gpu_fn, a_cpu, b_cpu, n_cpu, sync_gpu=True):
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
        out_gpu = gpu_fn(a_gpu, b_cpu, n_gpu)
        if sync_gpu:
            cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        t_gpu = t1 - t0
        # verify
        np.testing.assert_array_equal(out_cpu, cp.asnumpy(out_gpu))
        results[name] = (t_cpu, t_gpu)

    # Level-1: vectorized
    bench_batch_times("naive_modular_multiplication",
                     cpu.naive_modular_multiplication,
                     gpu.naive_modular_multiplication,
                     a1, b1, N1)
    bench_batch_times("naive_modular_exponentiation",
                     cpu.naive_modular_exponentiation,
                     gpu.naive_modular_exponentiation,
                     a1, e1, N1)

    # Level-2: loop-based
    MUL_IT = 16
    EXP_IT = 8
    BASE_MUL = 1 << 32
    BASE_EXP = 1 << 8
    a2_mul = rng.integers(0, BASE_MUL, size=BATCH1, dtype=np.int64)
    b2_mul = rng.integers(0, MUL_IT, size=BATCH1, dtype=np.int64)
    a2_exp = rng.integers(0, BASE_EXP, size=BATCH1, dtype=np.int64)
    b2_exp = rng.integers(0, EXP_IT, size=BATCH1, dtype=np.int64)
    bench_batch_times("naive_modular_multiplication_loop",
                     cpu.naive_modular_multiplication_loop,
                     gpu.naive_modular_multiplication_loop,
                     a2_mul, b2_mul, N1)
    bench_batch_times("naive_modular_exponentiation_loop",
                     cpu.naive_modular_exponentiation_loop,
                     gpu.naive_modular_exponentiation_loop,
                     a2_exp, b2_exp, N1)
    bench_batch_times("square_and_multiply_modular_exponentiation",
                     cpu.square_and_multiply_modular_exponentiation,
                     gpu.square_and_multiply_modular_exponentiation,
                     a2_exp, b2_exp, N1)

    # Level-3/4: Karatsuba and Montgomery variants
    N3 = 2_147_483_647
    BATCH3 = batch_size
    rng2 = np.random.default_rng(99)
    a3 = rng2.integers(0, N3, size=BATCH3, dtype=np.int64)
    b3 = rng2.integers(0, N3, size=BATCH3, dtype=np.int64)
    r3, nd3 = cpu.montgomery_precomputation(N3)

    # Karatsuba
    bench_batch_times("karatsuba_multiplication",
                     lambda a, b, n: cpu.karatsuba_multiplication(a, b),
                     lambda a, b, n: gpu.karatsuba_multiplication_gpu(a, b),
                     a3, b3, N3)
    # Montgomery
    cpu_mont = lambda x,y,n: cpu.montgomery_modular_multiplication(x,y,n,r3,nd3)
    gpu_mont = lambda x,y,n: gpu.montgomery_modular_multiplication(x,y,n,r3,nd3)
    bench_batch_times("montgomery_modular_multiplication",
                     cpu_mont, gpu_mont,
                     a3, b3, N3)
    # Optimized Montgomery
    cpu_opt = lambda x,y,n: cpu.optimized_montgomery_modular_multiplication(x,y,n,r3,nd3)
    gpu_opt = lambda x,y,n: gpu.optimized_montgomery_modular_multiplication(x,y,n,r3,nd3)
    bench_batch_times("optimized_montgomery_modular_multiplication",
                     cpu_opt, gpu_opt,
                     a3, b3, N3)
    # Karatsuba-Montgomery
    cpu_karamont = lambda x,y,n: cpu.montgomery_with_karatsuba(x,y,n,r3,nd3)
    gpu_karamont = lambda x,y,n: gpu.montgomery_with_karatsuba_gpu(x,y,n,r3,nd3)
    bench_batch_times("montgomery_with_karatsuba",
                     cpu_karamont, gpu_karamont,
                     a3, b3, N3)

    # Level-1: vectorized (using a_mul, b_mul, N3 for direct comparison with Montgomery)
    a_mul = rng2.integers(0, N3, size=BATCH3, dtype=np.int64)
    b_mul = rng2.integers(0, N3, size=BATCH3, dtype=np.int64)
    bench_batch_times("naive_modular_multiplication_level3inputs",
                     cpu.naive_modular_multiplication,
                     gpu.naive_modular_multiplication,
                     a_mul, b_mul, N3)

    return results


def main():
    # If called as a script, run with default batch size
    run_all_benchmarks()


if __name__ == "__main__":
    main()