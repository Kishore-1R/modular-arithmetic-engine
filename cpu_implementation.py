import numpy as np
#import cupy as cp

### Level 1
def naive_modular_multiplication(a, b, n):
    return (a*b) % n

def naive_modular_exponentiation(a, b, n):
    return (a**b) % n

### Level 2
def naive_modular_multiplication_loop(a, b, n):
    result = 0
    for i in range(b):
        result = (a + result) % n
    return result

def naive_modular_exponentiation_loop(a, b, n):
    result = 1
    for i in range(b):
        result = (a * result) % n
    return result

def square_and_multiply_modular_exponentiation(a, b, n):
    result = 1
    a = a % n
    while b > 0:
        if b & 1 == 1:                #b % 2 == 1:
            result = (result * a) % n
        b >>= 1                       #b = b // 2
        a = (a * a) % n
    return result

### Level 3: Montgomery Approach
def montgomery_precomputation(n):
    r = 1 << (n.bit_length())
    #r_inv = pow(r, -1, n)
    n_dash = (-pow(n, -1, r)) % r
    return r, n_dash

def to_montgomery(a, n, r):
    return (a * r) % n

def from_montgomery(a, n, r):
    # or use montgomery_modular_multiplication(a, 1, n, r, n_dash)    
    return (a * pow(r, -1, n)) % n
 
def montgomery_modular_multiplication(a, b, n, r, n_dash):
    t = a * b
    m = (t * n_dash) % r
    u = (t + m * n) // r
    return u - n if u >= n else u

def optimized_montgomery_modular_multiplication(a, b, n, r, n_dash):
    t = a * b
    m = (t * n_dash) & (r - 1)  # Fast modulo when r is power of 2
    u = (t + m * n) >> (r.bit_length() - 1)  # Fast division if r = 2^k
    return u - n if u >= n else u



# def montgomery_modular_exponentiation(a, b, n, r, n_dash):
#     a = (a * r) % n
#     result = 1 * r % n
#     for i in range(b.bit_length()):
#         if (b >> i) & 1:
#             result = montgomery_modular_multiplication(result, a, n, r, n_dash)
#         a = montgomery_modular_multiplication(a, a, n, r, n_dash)
#     return montgomery_modular_multiplication(result, 1, n, r, n_dash)

# def modular_inverse(a, n): 
#     t, new_t = 0, 1
#     r, new_r = n, a
#     while new_r != 0:
#         quotient = r // new_r
#         t, new_t = new_t, t - quotient * new_t
#         r, new_r = new_r, r - quotient * new_r
#     if r > 1:
#         raise ValueError("a is not invertible")
#     if t < 0:
#         t += n
#     return t


## GPU version with cupy

# def gpu_naive_modular_multiplication(a, b, n):
#     a = cp.asarray(a)
#     b = cp.asarray(b)
#     n = cp.asarray(n)
#     return (a*b) % n     