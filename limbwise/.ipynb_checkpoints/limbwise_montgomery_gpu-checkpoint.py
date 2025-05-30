# Example inputs
import numpy as np
import cupy as cp

# Montgomery multiplication using limb-wise operations on the GPU

def to_limb_batch(int_list, num_limbs=32):
    batch = []
    for x in int_list:
        limbs = [(x >> (32 * i)) & 0xFFFFFFFF for i in range(num_limbs)]
        batch.append(limbs)
    return cp.array(batch, dtype=cp.uint32)

def batch_montgomery_multiplication(A, B, N, n_prime):
    batch_size, num_limbs = A.shape
    T = cp.zeros((batch_size, 2 * num_limbs), dtype=cp.uint64)

    # Step 1: Schoolbook multiply A * B
    for i in range(num_limbs):
        for j in range(num_limbs):
            T[:, i + j] += A[:, i].astype(cp.uint64) * B[:, j].astype(cp.uint64)

    # Step 2: Montgomery reduction
    for i in range(num_limbs):
        m = (T[:, i] * n_prime) & 0xFFFFFFFF
        for j in range(num_limbs):
            T[:, i + j] += m * N[j]
        T[:, i + num_limbs] += T[:, i + num_limbs - 1] >> 32
        T[:, i + num_limbs - 1] &= 0xFFFFFFFF

    # Final reduction step
    result = T[:, num_limbs:2 * num_limbs].astype(cp.uint32)

    # Optional: conditional subtraction
    result_int = result.dot(1 << cp.arange(num_limbs * 32, step=32))  # convert limbs to int
    N_int = int("".join([f"{x:08x}" for x in reversed(N.tolist())]), 16)
    result_int = cp.where(result_int >= N_int, result_int - N_int, result_int)

    return result_int
