import numpy as np

### Level 1: purely array‐based (NumPy ufuncs)

def naive_modular_multiplication(a, b, n):
    """
    (a * b) % n
    Accepts NumPy scalars or arrays; broadcasts over inputs.
    """
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    return (a_arr * b_arr) % n


def naive_modular_exponentiation(a, b, n):
    """
    (a**b) % n
    Accepts NumPy scalars or arrays; broadcasts over inputs.
    Note: uses NumPy's vectorized power and mod.
    """
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    # np.power may produce huge intermediates—so we mod afterwards
    return np.mod(np.power(a_arr, b_arr), n)


### Level 2: Python‐loop scalars lifted to elementwise array versions

def naive_modular_multiplication_loop(a, b, n):
    """
    For each element (ai, bi), compute (ai added bi times) mod n.
    Works on scalars or arrays; returns an array of the broadcast shape.
    """
    # broadcast inputs to a common shape
    a_arr, b_arr = np.broadcast_arrays(a, b)
    # we'll store results in a same‐shaped int64 array
    out = np.empty(a_arr.shape, dtype=np.int64)

    # flatten to a 1D view for simple looping
    a_flat = a_arr.ravel()
    b_flat = b_arr.ravel()
    out_flat = out.ravel()

    for idx in range(a_flat.size):
        ai = int(a_flat[idx])
        bi = int(b_flat[idx])
        result = 0
        for _ in range(bi):
            result = (result + ai) % n
        out_flat[idx] = result

    # reshape back
    return out.reshape(a_arr.shape)


def naive_modular_exponentiation_loop(a, b, n):
    """
    For each element (ai, bi), compute pow(ai, bi) % n by repeated multiplication.
    """
    a_arr, b_arr = np.broadcast_arrays(a, b)
    out = np.empty(a_arr.shape, dtype=np.int64)

    a_flat = a_arr.ravel()
    b_flat = b_arr.ravel()
    out_flat = out.ravel()

    for idx in range(a_flat.size):
        ai = int(a_flat[idx])
        bi = int(b_flat[idx])
        result = 1
        for _ in range(bi):
            result = (result * ai) % n
        out_flat[idx] = result

    return out.reshape(a_arr.shape)


def square_and_multiply_modular_exponentiation(a, b, n):
    """
    For each element (ai, bi), compute ai**bi % n via square‐and‐multiply.
    """
    a_arr, b_arr = np.broadcast_arrays(a, b)
    out = np.empty(a_arr.shape, dtype=np.int64)

    a_flat = a_arr.ravel()
    b_flat = b_arr.ravel()
    out_flat = out.ravel()

    for idx in range(a_flat.size):
        ai = int(a_flat[idx]) % n
        bi = int(b_flat[idx])
        result = 1
        base = ai
        exp = bi
        while exp > 0:
            if exp & 1:
                result = (result * base) % n
            base = (base * base) % n
            exp >>= 1
        out_flat[idx] = result

    return out.reshape(a_arr.shape)


### Level 3: Montgomery Approach with vectorized arithmetic where possible

def montgomery_precomputation(n):
    """
    CPU‐only: returns Python ints (r, n_dash).
    """
    k = n.bit_length()
    r = 1 << k
    n_dash = (-pow(n, -1, r)) % r
    return r, n_dash


def to_montgomery(a, n, r):
    """
    Convert a to Montgomery form: (a * r) % n
    Works on scalars or arrays.
    """
    a_arr = np.asarray(a)
    return (a_arr * r) % n


def from_montgomery(a, n, r):
    """
    Convert a from Montgomery form: (a * r^{-1}) % n
    Works on scalars or arrays.
    """
    a_arr = np.asarray(a)
    # compute r^{-1} mod n once
    r_inv = pow(r, -1, n)
    return (a_arr * r_inv) % n


def montgomery_modular_multiplication(a, b, n, r, n_dash):
    """
    Classical REDC elementwise:
        t = a*b
        m = (t * n_dash) % r
        u = (t + m*n) // r
        if u>=n: u-=n
    Works on scalars or arrays using NumPy vector ops.
    """
    a_arr = np.asarray(a, dtype=np.int64)
    b_arr = np.asarray(b, dtype=np.int64)

    t = a_arr * b_arr
    m = (t * n_dash) % r
    u = (t + m * n) // r
    # conditional subtract
    return np.where(u >= n, u - n, u)


def optimized_montgomery_modular_multiplication(a, b, n, r, n_dash):
    """
    Fast REDC for r=2^k, elementwise:
        m = (a*b * n_dash) & (r-1)
        u = (a*b + m*n) >> k
        if u>=n: u-=n
    Uses NumPy bitwise ops.
    """
    a_arr = np.asarray(a, dtype=np.int64)
    b_arr = np.asarray(b, dtype=np.int64)

    k    = r.bit_length() - 1
    mask = (1 << k) - 1

    t = a_arr * b_arr
    m = (t * n_dash) & mask
    u = (t + m * n) >> k
    return np.where(u >= n, u - n, u)


def karatsuba_multiplication(a, b):
    """
    Karatsuba multiplication for two Python integers or NumPy arrays of integers.
    Accepts scalars or arrays; broadcasts over inputs.
    """
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    # Use Python int for large numbers
    def karatsuba(x, y):
        # Base case for small numbers
        if x.bit_length() <= 32 or y.bit_length() <= 32:
            return x * y
        n = max(x.bit_length(), y.bit_length())
        m = n // 2
        mask = (1 << m) - 1
        x_low = x & mask
        x_high = x >> m
        y_low = y & mask
        y_high = y >> m
        z0 = karatsuba(x_low, y_low)
        z2 = karatsuba(x_high, y_high)
        z1 = karatsuba(x_low + x_high, y_low + y_high) - z2 - z0
        return (z2 << (2 * m)) + (z1 << m) + z0
    # Vectorized version
    if a_arr.shape == () and b_arr.shape == ():
        return karatsuba(int(a_arr), int(b_arr))
    # Broadcast arrays
    a_flat = a_arr.ravel()
    b_flat = b_arr.ravel()
    out = np.empty_like(a_flat, dtype=object)
    for idx in range(a_flat.size):
        out[idx] = karatsuba(int(a_flat[idx]), int(b_flat[idx]))
    return out.reshape(a_arr.shape)


def montgomery_with_karatsuba(a, b, n, r, n_dash):
    """
    Optimized Montgomery multiplication using Karatsuba for the initial multiplication step.
    Fast REDC for r=2^k, elementwise:
        t = karatsuba(a, b)
        m = (t * n_dash) & (r-1)
        u = (t + m*n) >> k
        if u>=n: u-=n
    Uses NumPy bitwise ops and Python int for large numbers.
    """
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    k    = r.bit_length() - 1
    mask = (1 << k) - 1
    # Karatsuba multiplication (returns object dtype array)
    t = karatsuba_multiplication(a_arr, b_arr)
    # Convert n_dash and n to object arrays for broadcasting
    n_dash_arr = np.broadcast_to(n_dash, t.shape)
    n_arr = np.broadcast_to(n, t.shape)
    m = (t * n_dash_arr) & mask
    u = (t + m * n_arr) >> k
    # Conditional subtraction
    return np.where(u >= n_arr, u - n_arr, u)