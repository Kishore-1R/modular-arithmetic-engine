import numpy as np
import time
from RSA_Functions import montgomery_batch_cpu
from RSA_GPU import montgomery_powmod_gpu
import cupy as cp

def benchmark():
    n = 3233
    e = 17
    BATCH = 10000

    messages = np.random.randint(1, n, size=BATCH, dtype=np.uint64)

    # CPU
    t0 = time.perf_counter()
    cpu_result = montgomery_batch_cpu(messages, e, n)
    t1 = time.perf_counter()
    cpu_time = t1 - t0

    # GPU
    t0 = time.perf_counter()
    gpu_result = montgomery_powmod_gpu(messages, e, n)
    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()
    gpu_time = t1 - t0

    print(f"CPU time: {cpu_time:.4f} s")
    print(f"GPU time: {gpu_time:.4f} s")
    print(f"Speedup:  {cpu_time / gpu_time:.2f}x")

    np.testing.assert_array_equal(np.array(cpu_result, dtype=np.uint64), cp.asnumpy(gpu_result))

if __name__ == "__main__":
    benchmark()
