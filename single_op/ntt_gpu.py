import gpu_implementation as gpu
import numpy as np
# a: array of polynomial coefficients
# p: prime
# g: primitive root
def ntt_gpu(a, p, g):
    n = len(a)
    if n == 1:
        return a

    g_mod_p = gpu.naive_modular_multiplication(g, g, p)
    even = ntt_gpu(a[::2], p, g_mod_p)
    odd = ntt_gpu(a[1::2], p, g_mod_p)

    factor = 1;
    result = np.zeros(n, dtype=object)

    for i in range(n // 2):
        term = (factor * odd[i]) % p
        result[i] = (even[i] + term) % p
        result[i + n // 2] = (even[i] - term) % p
        factor = (factor * g_mod_p) % p
        
    return result

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
    result = np.zeros(n, dtype=object)

    for i in range(n // 2):
        term = (factor * odd[i]) % p
        result[i] = (even[i] + term) % p
        result[i + n // 2] = (even[i] - term) % p
        factor = (factor * g_mod_p) % p
        
    return result


def main():
    input_array = np.array([2, 4, 9, 4])
    prime = 17
    primitive_root = 3
    ntt_output = ntt_gpu_opt(input_array, prime, primitive_root)
    print("NTT:", ntt_output)
    intt_output = intt_gpu_opt(ntt_output, prime, primitive_root)
    print("INTT:", intt_output)
    
if __name__=="__main__":
    main()