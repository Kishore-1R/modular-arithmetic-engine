import time
import random
import numpy as np

# import your CPU and GPU modules
import cpu_implementation as cpu
import gpu_implementation as gpu
import ntt_gpu as ntt_gpu
import ntt_cpu as ntt_cpu

def time_function(func, args, repeat, sync_gpu=False):
    """
    Time `func(*args)` called `repeat` times.
    If sync_gpu=True, does a cp.cuda.Stream.null.synchronize() after the loop.
    Returns total elapsed seconds.
    """
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = func(*args)
    if sync_gpu:
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter() 
    return t1 - t0

def main():
    # choose a large prime modulus (61-bit Mersenne)
    N = 2305843009213693951  

    # random operands in [0, N)
    a = random.randrange(N)
    b = random.randrange(N)

    a_ntt = np.array([15855066,17545046,13244418,19205212,15029570,12057533,11317568,19170781,10774776,11086278,18879902,17143861,19926379,14150914,16731560,15070006,15965981,15632563,11059058,11412600])
    prime = 2305843009213693951
    primitive_root = 3
    r_ntt, n_dash_ntt = gpu.montgomery_precomputation(prime)

    # for exponentiation loops we pick a smaller exponent
    b_loop = 1            # small so naive loop finishes in reasonable time
    b_fm = 10**5             # for square‐and‐multiply

    # Montgomery precompute (on CPU)
    r, n_dash = cpu.montgomery_precomputation(N)

    

    # Warm‐up the GPU once so kernels compile before timing
    print("Warming up GPU kernels...")
    _ = gpu.naive_modular_multiplication(a, b, N)
    _ = gpu.naive_modular_exponentiation(a, b, N)
    _ = gpu.naive_modular_multiplication_loop(a, b_loop, N)
    _ = gpu.naive_modular_exponentiation_loop(a, b_loop, N)
    _ = gpu.square_and_multiply_modular_exponentiation(a, b_fm, N)
    _ = gpu.montgomery_modular_multiplication(a, b, N, r, n_dash)
    _ = gpu.optimized_montgomery_modular_multiplication(a, b, N, r, n_dash)
    import cupy as cp; cp.cuda.Stream.null.synchronize()
    print("Done warm-up.\n")

    benchmarks = [
        # (label, cpu_func, gpu_func, args_cpu, args_gpu, cpu_reps, gpu_reps, sync_gpu)
        ("naive_mul", 
            cpu.naive_modular_multiplication, 
            gpu.naive_modular_multiplication, 
            (a, b, N), (a, b, N), 
            10_000, 10_000, False),

        ("naive_exp", 
            cpu.naive_modular_exponentiation, 
            gpu.naive_modular_exponentiation, 
            (a, b_fm, N), (a, b_fm, N), 
            1, 1, False),

        ("loop_mul", 
            cpu.naive_modular_multiplication_loop, 
            gpu.naive_modular_multiplication_loop, 
            (a, b_loop, N), (a, b_loop, N), 
            2_00, 2_00, True),

        ("loop_exp", 
            cpu.naive_modular_exponentiation_loop, 
            gpu.naive_modular_exponentiation_loop, 
            (a, b_loop, N), (a, b_loop, N), 
            200, 200, True),

        ("sqr_mul_exp", 
            cpu.square_and_multiply_modular_exponentiation, 
            gpu.square_and_multiply_modular_exponentiation, 
            (a, b_fm, N), (a, b_fm, N), 
            10_00, 10_00, True),

        ("montgomery_mul", 
            lambda x,y: cpu.montgomery_modular_multiplication(x, y, N, r, n_dash),
            lambda x,y: gpu.montgomery_modular_multiplication(x, y, N, r, n_dash),
            (a, b), (a, b), 
            10_00, 10_00, True),

        ("opt_mont_mul",
            lambda x,y: cpu.optimized_montgomery_modular_multiplication(x, y, N, r, n_dash),
            lambda x,y: gpu.optimized_montgomery_modular_multiplication(x, y, N, r, n_dash),
            (a, b), (a, b),
            10_00, 10_00, True),
        ("ntt_naive",
            lambda x,y: ntt_cpu.ntt_cpu(a_ntt, prime, primitive_root),
            lambda x,y: ntt_gpu.ntt_gpu(a_ntt, prime, primitive_root),
            (a, b), (a, b),
            10, 1, True),
        ("ntt_mont",
            lambda x,y: ntt_cpu.ntt_cpu_mont(a_ntt, prime, primitive_root, r_ntt, n_dash_ntt),
            lambda x,y: ntt_gpu.ntt_gpu_opt(a_ntt, prime, primitive_root),
            (a, b), (a, b),
            10, 1, True),
    ]

    print(f"{'Kernel':20s}  {'CPU time (s)':>12s}   {'GPU time (s)':>12s}   {'speedup':>8s}")
    print("-" * 60)

    for label, cpu_f, gpu_f, args_c, args_g, reps_c, reps_g, sync in benchmarks:
        # CPU timing
        t_cpu = time_function(cpu_f, args_c, reps_c, sync_gpu=False)
        # GPU timing (with sync at end if needed)
        t_gpu = time_function(gpu_f, args_g, reps_g, sync_gpu=sync)
        speed = t_cpu / t_gpu if t_gpu>0 else float('inf')
        print(f"{label:20s}   {t_cpu:12.4f}   {t_gpu:12.4f}   {speed:8.2f}x")

if __name__ == "__main__":
    main()