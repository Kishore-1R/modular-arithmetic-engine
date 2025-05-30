import time
import random
import numpy as np
import cupy as cp

# Import your implementations
import cpu_implementation as cpu
import limbwise_montgomery_gpu as gpu  # Assumes batch_montgomery_modular_multiplication is defined

def time_function(func, args, repeat, sync_gpu=False):
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = func(*args)
    if sync_gpu:
        cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()
    return t1 - t0

def main():
    # 61-bit Mersenne prime
    N = 2305843009213693951
    r, n_dash = cpu.montgomery_precomputation(N)

    # Batch size for testing GPU speedup
    batch_size = 10_000

    # Generate random input arrays
    batch_a = np.random.randint(0, N, size=batch_size, dtype=object)
    batch_b = np.random.randint(0, N, size=batch_size, dtype=object)

    # Warm-up GPU kernel
    print("Warming up GPU...")
    batch_a_cp = cp.asarray(batch_a, dtype=object)
    batch_b_cp = cp.asarray(batch_b, dtype=object)
    _ = gpu.batch_montgomery_modular_multiplication(batch_a_cp, batch_b_cp, N, r, n_dash)
    cp.cuda.Stream.null.synchronize()
    print("Warm-up done.\n")

    # CPU batch timing (scalar loop)
    def cpu_batch_montgomery_mul(batch_a, batch_b):
        return [cpu.montgomery_modular_multiplication(a, b, N, r, n_dash)
                for a, b in zip(batch_a, batch_b)]

    # GPU batch timing
    def gpu_batch_montgomery_mul(batch_a_cp, batch_b_cp):
        return gpu.batch_montgomery_modular_multiplication(batch_a_cp, batch_b_cp, N, r, n_dash)

    print(f"{'Kernel':30s}  {'CPU time (s)':>12s}   {'GPU time (s)':>12s}   {'Speedup':>8s}")
    print("-" * 70)

    # Benchmark CPU
    t_cpu = time_function(cpu_batch_montgomery_mul, (batch_a, batch_b), repeat=1)
    
    # Benchmark GPU
    t_gpu = time_function(gpu_batch_montgomery_mul, (batch_a_cp, batch_b_cp), repeat=1, sync_gpu=True)

    # Print results
    print(f"{'batch_montgomery_mul':30s}   {t_cpu:12.4f}   {t_gpu:12.4f}   {t_cpu / t_gpu:8.2f}x")

if __name__ == "__main__":
    main()
