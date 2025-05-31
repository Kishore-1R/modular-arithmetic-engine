def montgomery_precompute(n):
    k = n.bit_length()
    r = 1 << k
    r_inv = pow(r, -1, n)
    n_dash = (-pow(n, -1, r)) % r
    return r, r_inv, n_dash, k

def montgomery_reduce(t, n, r, n_dash):
    m = (t * n_dash) % r
    u = (t + m * n) // r
    return u - n if u >= n else u

def montgomery_multiply(a, b, n, r, n_dash):
    return montgomery_reduce(a * b, n, r, n_dash)

def montgomery_powmod(base, exp, n):
    r, r_inv, n_dash, _ = montgomery_precompute(n)
    base = (base * r) % n
    result = (1 * r) % n

    for bit in bin(exp)[2:]:
        result = montgomery_multiply(result, result, n, r, n_dash)
        if bit == '1':
            result = montgomery_multiply(result, base, n, r, n_dash)

    return montgomery_multiply(result, 1, n, r, n_dash)

def montgomery_batch_cpu(messages, exp, n):
    return [montgomery_powmod(m, exp, n) for m in messages]