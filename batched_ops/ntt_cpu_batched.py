import cpu_implementation_batched as cpu
import numpy as np
# a: array of polynomial coefficients
# p: prime
# g: primitive root
def ntt_cpu(a, p, g):
    n = len(a)
    if n == 1:
        return a

    g_mod_p = cpu.naive_modular_multiplication(g, g, p)
    even = ntt_cpu(a[::2], p, g_mod_p)
    odd = ntt_cpu(a[1::2], p, g_mod_p)

    factor = 1;
    result = np.zeros(n, dtype=np.longlong)

    for i in range(n // 2):
        #term = (factor * odd[i]) % p
        term = cpu.naive_modular_multiplication(factor, odd[i], p)
        result[i] = (even[i] + term) % p
        result[i + n // 2] = (even[i] - term) % p
        #factor = (factor * g_mod_p) % p
        factor = cpu.naive_modular_multiplication(factor, g_mod_p, p)
        
    return result


def ntt_cpu_opt(a, p, g, r, n_dash, factor_mont):
    n = len(a)
    if n == 1:
        return a
        
    g_mod_p = cpu.optimized_montgomery_modular_multiplication(g, g, p, r, n_dash)
    even = ntt_cpu_opt(a[::2], p, g_mod_p, r, n_dash, factor_mont)
    odd = ntt_cpu_opt(a[1::2], p, g_mod_p, r, n_dash, factor_mont)

    # factor = cpu.to_montgomery(1, p, r)
    result = np.zeros(n, dtype=	np.longlong)
    factor = factor_mont
    for i in range(n // 2):
        term = cpu.optimized_montgomery_modular_multiplication(factor, odd[i], p, r, n_dash)
        result[i] = (even[i] + term) % p
        result[i + n // 2] = (even[i] - term + p) % p
        factor = cpu.optimized_montgomery_modular_multiplication(factor, g_mod_p, p, r, n_dash)
        
    return result


def ntt_cpu_mont(a, p, g, r, n_dash):
    n = len(a)

    a_mont = cpu.to_montgomery(a, p, r)
    g_mont = cpu.to_montgomery(g, p, r)
    factor_mont = cpu.to_montgomery(1, p, r)
    out_mont = ntt_cpu_opt(a_mont, p, g_mont, r, n_dash, factor_mont)

    result = cpu.from_montgomery(out_mont, p, r)
    return result


def main():
    input_array = np.array([2, 4, 9, 4])
    prime = 17
    primitive_root = 3
    r, n_dash = cpu.montgomery_precomputation(prime)
    ntt_output = ntt_cpu_mont(input_array, prime, primitive_root, r, n_dash)
    print("NTT:", ntt_output)
    
if __name__=="__main__":
    main()