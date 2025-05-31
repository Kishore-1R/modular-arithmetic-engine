import cupy as cp
from RSA_Functions import montgomery_precompute

def montgomery_powmod_gpu(messages, exp, n):
    r, r_inv, n_dash, _ = montgomery_precompute(n)

    messages = cp.asarray(messages, dtype=cp.uint64)
    n = cp.uint64(n)
    r = cp.uint64(r)
    n_dash = cp.uint64(n_dash)

    def gpu_scalar(base):
        base = (base * r) % n
        result = (1 * r) % n
        for bit in bin(exp)[2:]:
            result = (result * result) % n
            if bit == '1':
                result = (result * base) % n
        return (result * 1) % n

    return cp.asarray([gpu_scalar(m) for m in messages])