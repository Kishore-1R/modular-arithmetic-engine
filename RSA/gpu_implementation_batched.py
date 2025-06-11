import cupy as cp

### Level 1
def naive_modular_multiplication(a, b, n):
    """
    Computes (a*b) % n on the GPU.
    a, b, n can be scalars or broadcastable arrays.
    """
    a = cp.asarray(a)
    b = cp.asarray(b)
    n = cp.asarray(n)
    return (a * b) % n

def naive_modular_exponentiation(a, b, n):
    """
    Computes (a**b) % n on the GPU.
    Uses cupy.power under the hood.
    """
    a = cp.asarray(a)
    b = cp.asarray(b)
    n = cp.asarray(n)
    return cp.power(a, b) % n

### Level 2

# Kernel for naive modular multiplication by repeated addition
_naive_mod_mul_loop_kernel = cp.ElementwiseKernel(
    in_params='uint64 a, uint64 b, uint64 n',
    out_params='uint64 out',
    operation=r'''
        unsigned long long result = 0;
        for (unsigned long long i = 0; i < b; ++i) {
            result = (a + result) % n;
        }
        out = result;
    ''',
    name='naive_mod_mul_loop'
)

def naive_modular_multiplication_loop(a, b, n):
    """
    Computes (a*b) % n by looping b times, entirely on the GPU.
    """
    a = cp.asarray(a, dtype=cp.uint64)
    b = cp.asarray(b, dtype=cp.uint64)
    n = cp.asarray(n, dtype=cp.uint64)
    return _naive_mod_mul_loop_kernel(a, b, n)


# Kernel for naive modular exponentiation by repeated multiplication
_naive_mod_exp_loop_kernel = cp.ElementwiseKernel(
    in_params='uint64 a, uint64 b, uint64 n',
    out_params='uint64 out',
    operation=r'''
        unsigned long long result = 1;
        for (unsigned long long i = 0; i < b; ++i) {
            result = (result * a) % n;
        }
        out = result;
    ''',
    name='naive_mod_exp_loop'
)

def naive_modular_exponentiation_loop(a, b, n):
    """
    Computes (a**b) % n by looping b times, entirely on the GPU.
    """
    a = cp.asarray(a, dtype=cp.uint64)
    b = cp.asarray(b, dtype=cp.uint64)
    n = cp.asarray(n, dtype=cp.uint64)
    return _naive_mod_exp_loop_kernel(a, b, n)


# Kernel for square‐and‐multiply modular exponentiation
_square_and_multiply_kernel = cp.ElementwiseKernel(
    in_params='uint64 a, uint64 b, uint64 n',
    out_params='uint64 out',
    operation=r'''
        unsigned long long result = 1;
        unsigned long long base   = a % n;
        unsigned long long exp    = b;
        while (exp > 0) {
            if (exp & 1ULL) {
                result = (result * base) % n;
            }
            base = (base * base) % n;
            exp >>= 1;
        }
        out = result;
    ''',
    name='square_and_multiply'
)

def square_and_multiply_modular_exponentiation(a, b, n):
    """
    Fast (a**b) % n using square_and_multiply, entirely on the GPU.
    """
    a = cp.asarray(a, dtype=cp.uint64)
    b = cp.asarray(b, dtype=cp.uint64)
    n = cp.asarray(n, dtype=cp.uint64)
    return _square_and_multiply_kernel(a, b, n)


#
# Level 3: Montgomery Approach
#

def montgomery_precomputation(n):
    """
    Compute (r, n_dash) for Montgomery multiplication.
    Returns Python ints.  These are small and not worth pushing to the GPU.
    """
    # choose r = 2^k > n
    k = n.bit_length()
    r = 1 << k
    # n_dash = -n^{-1} mod r
    n_dash = (-pow(n, -1, r)) % r
    return r, n_dash


def to_montgomery(a, n, r):
    """
    Convert a into Montgomery form:  a * r mod n
    Runs on the GPU via simple broadcasted ops.
    """
    a = cp.asarray(a, dtype=cp.uint64)
    n = cp.asarray(n, dtype=cp.uint64)
    r = cp.asarray(r, dtype=cp.uint64)
    return (a * r) % n


def from_montgomery(a, n, r):
    """
    Convert a out of Montgomery form: a * r^{-1} mod n
    We compute r^{-1} on the CPU (cheap) then do a GPU multiply+mod.
    """
    a = cp.asarray(a, dtype=cp.uint64)
    n_int = int(n)
    r_int = int(r)
    r_inv = pow(r_int, -1, n_int)
    r_inv = cp.uint64(r_inv)
    n = cp.asarray(n_int, dtype=cp.uint64)
    return (a * r_inv) % n


# Kernel for classical Montgomery modular multiplication
_montgomery_mul_kernel = cp.ElementwiseKernel(
    in_params=(
        'uint64 a, uint64 b, '
        'uint64 n, uint64 r, uint64 n_dash'
    ),
    out_params='uint64 out',
    operation=r'''
        // t = a * b
        unsigned long long t = a * b;
        // m = (t * n_dash) mod r
        unsigned long long m = (t * n_dash) % r;
        // u = (t + m * n) / r
        unsigned long long u = (t + m * n) / r;
        // if u >= n, subtract n
        if (u >= n) u -= n;
        out = u;
    ''',
    name='montgomery_mul'
)

def montgomery_modular_multiplication(a, b, n, r, n_dash):
    """
    Montgomery modular multiplication on the GPU.
    All inputs can be scalars or arrays.
    """
    a = cp.asarray(a, dtype=cp.uint64)
    b = cp.asarray(b, dtype=cp.uint64)
    n = cp.asarray(n, dtype=cp.uint64)
    r = cp.asarray(r, dtype=cp.uint64)
    n_dash = cp.asarray(n_dash, dtype=cp.uint64)
    return _montgomery_mul_kernel(a, b, n, r, n_dash)


# Kernel for optimized Montgomery when r=2^k
_optimized_montgomery_mul_kernel = cp.ElementwiseKernel(
    in_params=(
        'uint64 a, uint64 b, '
        'uint64 n, uint64 r_mask, uint64 shift, uint64 n_dash'
    ),
    out_params='uint64 out',
    operation=r'''
        unsigned long long t = a * b;
        // fast mod r via bitmask
        unsigned long long m = (t * n_dash) & r_mask;
        // fast div r via right shift
        unsigned long long u = (t + m * n) >> shift;
        if (u >= n) u -= n;
        out = u;
    ''',
    name='optimized_montgomery_mul'
)

def optimized_montgomery_modular_multiplication(a, b, n, r, n_dash):
    """
    Montgomery multiplication optimized for r = 2^k.
    r and n_dash must be Python ints (as returned by montgomery_precomputation).
    """
    # 1) compute mask and shift on the CPU
    #    r = 1 << k  =>  k = bit_length(r)-1
    k    = r.bit_length() - 1
    mask = (1 << k) - 1

    # 2) move data to GPU-friendly uint64 arrays/scalars
    a_cpu      = int(a) if isinstance(a, (int,)) else None
    b_cpu      = int(b) if isinstance(b, (int,)) else None
    n_cpu      = int(n) if isinstance(n, (int,)) else None

    a_gpu      = cp.asarray(a_cpu if a_cpu is not None else a, dtype=cp.uint64)
    b_gpu      = cp.asarray(b_cpu if b_cpu is not None else b, dtype=cp.uint64)
    n_gpu      = cp.asarray(n_cpu if n_cpu is not None else n, dtype=cp.uint64)
    r_mask_gpu = cp.uint64(mask)
    shift_gpu  = cp.uint64(k)
    n_dash_gpu = cp.asarray(n_dash, dtype=cp.uint64)

    # 3) launch the elementwise kernel
    return _optimized_montgomery_mul_kernel(
        a_gpu, b_gpu, n_gpu, r_mask_gpu, shift_gpu, n_dash_gpu
    )


# Kernel for elementwise Karatsuba multiplication (recursive, but only for small 32-bit values)
_karatsuba_kernel = cp.ElementwiseKernel(
    in_params='uint64 a, uint64 b',
    out_params='uint64 out',
    operation=r'''
        // Only for small values (32 bits or less)
        unsigned long long x = a;
        unsigned long long y = b;
        if (x < (1ULL << 32) && y < (1ULL << 32)) {
            // Use Karatsuba for 32-bit values
            unsigned int m = 16;
            unsigned int mask = (1U << m) - 1U;
            unsigned int x0 = x & mask;
            unsigned int x1 = x >> m;
            unsigned int y0 = y & mask;
            unsigned int y1 = y >> m;
            unsigned long long z0 = (unsigned long long)x0 * y0;
            unsigned long long z2 = (unsigned long long)x1 * y1;
            unsigned long long z1 = (unsigned long long)(x0 + x1) * (y0 + y1) - z2 - z0;
            out = (z2 << (2 * m)) + (z1 << m) + z0;
        } else {
            // Fallback to schoolbook for larger values
            out = x * y;
        }
    ''',
    name='karatsuba_mul_gpu'
)

def karatsuba_multiplication_gpu(a, b):
    """
    Elementwise Karatsuba multiplication for two cupy arrays or scalars (on GPU for 32-bit values, schoolbook for larger).
    """
    a_cp = cp.asarray(a, dtype=cp.uint64)
    b_cp = cp.asarray(b, dtype=cp.uint64)
    return _karatsuba_kernel(a_cp, b_cp)


def montgomery_with_karatsuba_gpu(a, b, n, r, n_dash):
    """
    Optimized Montgomery multiplication using Karatsuba for the initial multiplication step (on GPU for 32-bit values, schoolbook for larger).
    Fast REDC for r=2^k, elementwise:
        t = karatsuba(a, b)
        m = (t * n_dash) & (r-1)
        u = (t + m*n) >> k
        if u>=n: u-=n
    """
    a_cp = cp.asarray(a)
    b_cp = cp.asarray(b)
    n_cp = cp.asarray(n)
    # Compute mask and shift
    k = int(r).bit_length() - 1
    mask = (1 << k) - 1
    mask_cp = cp.asarray(mask, dtype=cp.uint64)
    # Ensure k_cp is the same dtype as t (avoid float64 bug)
    k_cp = cp.array(k, dtype=cp.uint64)
    # Karatsuba multiplication (elementwise)
    t = karatsuba_multiplication_gpu(a_cp, b_cp)
    n_dash_cp = cp.asarray(n_dash, dtype=cp.uint64)
    n_cp_b = cp.broadcast_to(n_cp.astype(cp.uint64), t.shape)
    m = (t * n_dash_cp) & mask_cp
    u = (t + m * n_cp_b) >> k_cp
    return cp.where(u >= n_cp_b, u - n_cp_b, u)