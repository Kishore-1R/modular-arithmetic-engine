import cpu_implementation as cpu
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
    result = np.zeros(n, dtype=int)

    for i in range(n // 2):
        term = (factor * odd[i]) % p
        result[i] = (even[i] + term) % p
        result[i + n // 2] = (even[i] - term) % p
        factor = (factor * g_mod_p) % p
        
    return result

def intt_cpu_rec(a, p, g):
    n = len(a)
    if n == 1:
        return a

    g_inv = pow(g, -1, p)

    g_inv_mod_p = cpu.naive_modular_multiplication(g_inv, g_inv, p)
    even = intt_cpu_rec(a[::2], p, g_inv_mod_p)
    odd = intt_cpu_rec(a[1::2], p, g_inv_mod_p)

    factor = 1
    result = np.zeros(n, dtype=int)

    for i in range(n // 2):
        term = (factor * odd[i]) % p
        result[i] = (even[i] + term) % p
        result[i + n // 2] = (even[i] - term) % p
        factor = (factor * g_inv_mod_p) % p
        
    return result

def intt_cpu(a, p, g):
    n = len(a)
    a = intt_cpu_rec(a, p, g)
    n_inv = pow(n, -1, p)
    
    return [cpu.naive_modular_multiplication(x, n_inv, p).item() for x in a]

def ntt_cpu_opt(a, p, g, r, n_dash):
    n = len(a)
    if n == 1:
        return a
        
    g_mod_p = cpu.optimized_montgomery_modular_multiplication(g, g, p, r, n_dash)
    even = ntt_cpu_opt(a[::2], p, g_mod_p, r, n_dash)
    odd = ntt_cpu_opt(a[1::2], p, g_mod_p, r, n_dash)

    factor = 1;
    result = np.zeros(n, dtype=int)

    for i in range(n // 2):
        term = (factor * odd[i]) % p
        result[i] = (even[i] + term) % p
        result[i + n // 2] = (even[i] - term) % p
        factor = (factor * g_mod_p) % p
        
    return result

def intt_cpu_rec_opt(a, p, g, r, n_dash):
    n = len(a)
    if n == 1:
        return a
    g_inv = pow(g, -1, p)
    #g_inv_mont = cpu.to_montgomery(g_inv, p, r)
    g_inv_mod_p = cpu.optimized_montgomery_modular_multiplication(g_inv, g_inv, p, r, n_dash)

    even = intt_cpu_rec_opt(a[::2], p, g_inv_mod_p, r, n_dash)
    odd = intt_cpu_rec_opt(a[1::2], p, g_inv_mod_p, r, n_dash)

    factor = 1
    result = np.zeros(n, dtype=int)

    for i in range(n // 2):
        term = (factor * odd[i]) % p
        result[i] = (even[i] + term) % p
        result[i + n // 2] = (even[i] - term) % p
        factor = (factor * g_inv_mod_p) % p
        
    return result

def intt_cpu_opt(a, p, g, r, n_dash):
    n = len(a)
    a = intt_cpu_rec_opt(a, p, g, r, n_dash)
    n_inv = pow(n, -1, p)
    r, n_dash = cpu.montgomery_precomputation(p)
    return [cpu.from_montgomery(cpu.optimized_montgomery_modular_multiplication(cpu.to_montgomery(x, p, r), cpu.to_montgomery(n_inv, p, r), p, r, n_dash), p, r).item() for x in a]


def main():
    input_array = np.array([2, 4, 9, 4])
    prime = 17
    primitive_root = 3
    r, n_dash = cpu.montgomery_precomputation(prime)
    ntt_output = ntt_cpu(input_array, prime, primitive_root)
    print("NTT:", ntt_output)
    intt_output = intt_cpu(ntt_output, prime, primitive_root)
    print("INTT:", intt_output)
    
if __name__=="__main__":
    main()