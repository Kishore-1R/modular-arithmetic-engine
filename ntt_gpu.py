import single_op.gpu_implementation as gpu
import numpy as np
# a: array of polynomial coefficients
# p: primitive root
# g: modulus
def ntt_gpu(a, p, g):
    n = len(a)
    if n == 1:
        return a

    g_mod_p = gpu.naive_modular_multiplication(g, g, p)
    even = ntt_gpu(a[::2], p, g_mod_p)
    odd = ntt_gpu(a[1::2], p, g_mod_p)

    factor = 1;
    result = np.zeros(n, dtype=int)

    for i in range(n // 2):
        term = (factor * odd[i]) % p
        result[i] = (even[i] + term) % p
        result[i + n // 2] = (even[i] - term) % p
        factor = (factor * g_mod_p) % p
        
    return result

def intt_gpu_rec(a, p, g):
    n = len(a)
    if n == 1:
        return a

    g_inv = pow(g, -1, p)

    g_inv_mod_p = gpu.naive_modular_multiplication(g_inv, g_inv, p)
    even = intt_gpu_rec(a[::2], p, g_inv_mod_p)
    odd = intt_gpu_rec(a[1::2], p, g_inv_mod_p)

    factor = 1
    result = np.zeros(n, dtype=int)

    for i in range(n // 2):
        term = (factor * odd[i]) % p
        result[i] = (even[i] + term) % p
        result[i + n // 2] = (even[i] - term) % p
        factor = (factor * g_inv_mod_p) % p
        
    return result

def intt_gpu(a, p, g):
    n = len(a)
    a = intt_gpu_rec(a, p, g)
    n_inv = pow(n, -1, p)
    
    return [gpu.naive_modular_multiplication(x, n_inv, p).item() for x in a]

def ntt_gpu_opt(a, p, g):
    n = len(a)
    if n == 1:
        return a
    r, n_dash = gpu.montgomery_precomputation(p)
    g_mont = gpu.to_montgomery(g, p, r)
    g_mod_p = gpu.from_montgomery(gpu.optimized_montgomery_modular_multiplication(g_mont, g_mont, p, r, n_dash), p, r)
    even = ntt_gpu(a[::2], p, g_mod_p)
    odd = ntt_gpu(a[1::2], p, g_mod_p)

    factor = 1;
    result = np.zeros(n, dtype=int)

    for i in range(n // 2):
        term = (factor * odd[i]) % p
        result[i] = (even[i] + term) % p
        result[i + n // 2] = (even[i] - term) % p
        factor = (factor * g_mod_p) % p
        
    return result

def intt_gpu_rec_opt(a, p, g):
    n = len(a)
    if n == 1:
        return a

    g_inv = pow(g, -1, p)

    r, n_dash = gpu.montgomery_precomputation(p)
    g_inv_mont = gpu.to_montgomery(g_inv, p, r)
    g_inv_mod_p = gpu.from_montgomery(gpu.optimized_montgomery_modular_multiplication(g_inv_mont, g_inv_mont, p, r, n_dash), p, r)
    even = intt_gpu_rec(a[::2], p, g_inv_mod_p)
    odd = intt_gpu_rec(a[1::2], p, g_inv_mod_p)

    factor = 1
    result = np.zeros(n, dtype=int)

    for i in range(n // 2):
        term = (factor * odd[i]) % p
        result[i] = (even[i] + term) % p
        result[i + n // 2] = (even[i] - term) % p
        factor = (factor * g_inv_mod_p) % p
        
    return result

def intt_gpu_opt(a, p, g):
    n = len(a)
    a = intt_gpu_rec_opt(a, p, g)
    n_inv = pow(n, -1, p)
    r, n_dash = gpu.montgomery_precomputation(p)
    return [gpu.from_montgomery(gpu.optimized_montgomery_modular_multiplication(gpu.to_montgomery(x, p, r), gpu.to_montgomery(n_inv, p, r), p, r, n_dash), p, r).item() for x in a]


def main():
    input_array = np.array([12, 20, 33, 14])
    prime = 37
    primitive_root = 2
    ntt_output = ntt_gpu_opt(input_array, prime, primitive_root)
    print("NTT:", ntt_output)
    intt_output = intt_gpu_opt(ntt_output, prime, primitive_root)
    print("INTT:", intt_output)
    
if __name__=="__main__":
    main()