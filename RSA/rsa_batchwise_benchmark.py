
import time
import random
from cpu_implementation_batched import (
    naive_modular_exponentiation,
    naive_modular_exponentiation_loop,
    square_and_multiply_modular_exponentiation,
    montgomery_precomputation,
    montgomery_modular_multiplication,
    optimized_montgomery_modular_multiplication,
    montgomery_with_karatsuba
)

def generate_rsa_test_vectors(batch_size=100, bits=1024):
    e = 65537
    messages = []
    exponents = []
    moduli = []

    for _ in range(batch_size):
        p = random.getrandbits(bits // 2)
        q = random.getrandbits(bits // 2)
        n = p * q
        m = random.randrange(2, n - 1)
        messages.append(m)
        exponents.append(e)
        moduli.append(n)

    return messages, exponents, moduli

def rsa_encrypt_with_method(messages, exponents, moduli, method, use_montgomery=False):
    start = time.time()
    results = []
    for m, e, n in zip(messages, exponents, moduli):
        if use_montgomery:
            r, n_dash = montgomery_precomputation(n)
            result = method(m, e, n, r, n_dash)
        else:
            result = method(m, e, n)
        results.append(result)
    end = time.time()
    return end - start

def run_all_benchmarks(batch_size):
    messages, exponents, moduli = generate_rsa_test_vectors(batch_size)

    timings = {}

    timings["naive_modular_exponentiation"] = (
        rsa_encrypt_with_method(messages, exponents, moduli, naive_modular_exponentiation),
        None
    )
    timings["naive_modular_exponentiation_loop"] = (
        rsa_encrypt_with_method(messages, exponents, moduli, naive_modular_exponentiation_loop),
        None
    )
    timings["square_and_multiply_modular_exponentiation"] = (
        rsa_encrypt_with_method(messages, exponents, moduli, square_and_multiply_modular_exponentiation),
        None
    )
    timings["optimized_montgomery_modular_multiplication"] = (
        rsa_encrypt_with_method(messages, exponents, moduli, optimized_montgomery_modular_modular_exponentiation, use_montgomery=True),
        None
    )
    timings["montgomery_with_karatsuba"] = (
        rsa_encrypt_with_method(messages, exponents, moduli, montgomery_with_karatsuba, use_montgomery=True),
        None
    )

    return timings
