import os
import subprocess
import re
import matplotlib.pyplot as plt
from batchwise_benchmark import run_all_benchmarks

# List of batch sizes to test
batch_sizes = [1000, 5000, 20000, 100000, 500000, 1000000, 2000000]

# Methods for each level
level1_methods = [
    "naive_modular_multiplication",
    "naive_modular_exponentiation"
]
level2_methods = [
    "naive_modular_multiplication_loop",
    "naive_modular_exponentiation_loop",
    "square_and_multiply_modular_exponentiation"
]
level3_methods = [
    "karatsuba_multiplication",
    "montgomery_modular_multiplication",
    "optimized_montgomery_modular_exponentiation",
    "montgomery_with_karatsuba"
]

# Store results: {level: {method: {batch_size: (cpu_time, gpu_time)}}}
results = {
    1: {m: {} for m in level1_methods},
    2: {m: {} for m in level2_methods},
    3: {m: {} for m in level3_methods},
}

for batch in batch_sizes:
    print(f"Running batch size {batch}...")
    timings = run_all_benchmarks(batch)
    for m in level1_methods:
        if m in timings:
            results[1][m][batch] = timings[m]
    for m in level2_methods:
        if m in timings:
            results[2][m][batch] = timings[m]
    for m in level3_methods:
        if m in timings:
            results[3][m][batch] = timings[m]

# Plotting function

def plot_level(level, methods, title, filename):
    plt.figure(figsize=(10, 6))
    for method in methods:
        if not results[level][method]:
            continue
        batches = sorted(results[level][method].keys())
        cpu_times = [results[level][method][b][0] for b in batches]
        gpu_times = [results[level][method][b][1] for b in batches]
        plt.plot(batches, cpu_times, marker='o', label=f"{method} (CPU)")
        plt.plot(batches, gpu_times, marker='x', label=f"{method} (GPU)")
    plt.xlabel("Batch size")
    plt.ylabel("Time (s)")
    plt.title(title)
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# # Plot and save each level separately
# plot_level(1, level1_methods, "Level 1: Vectorized Modular Arithmetic (CPU vs GPU)", "level1_vectorized.png")
# plot_level(2, level2_methods, "Level 2: Loop-based Modular Arithmetic (CPU vs GPU)", "level2_loop.png")
# plot_level(3, level3_methods, "Level 3/4: Karatsuba & Montgomery Variants (CPU vs GPU)", "level3_karatsuba_montgomery.png")

# Plot and save multiplication and exponentiation methods separately for each level
# Multiplication methods
level1_mul = ["naive_modular_multiplication"]
level2_mul = ["naive_modular_multiplication_loop"]
level3_mul = ["karatsuba_multiplication", "montgomery_modular_multiplication", "optimized_montgomery_modular_exponentiation", "montgomery_with_karatsuba"]
# Exponentiation methods
level1_exp = ["naive_modular_exponentiation"]
level2_exp = ["naive_modular_exponentiation_loop", "square_and_multiply_modular_exponentiation"]

plot_level(1, level1_mul, "Level 1: Multiplication (CPU vs GPU)", "level1_mul.png")
plot_level(1, level1_exp, "Level 1: Exponentiation (CPU vs GPU)", "level1_exp.png")
plot_level(2, level2_mul, "Level 2: Multiplication (CPU vs GPU)", "level2_mul.png")
plot_level(2, level2_exp, "Level 2: Exponentiation (CPU vs GPU)", "level2_exp.png")
plot_level(3, level3_mul, "Level 3/4: Multiplication (CPU vs GPU)", "level3_mul.png")
# (No exponentiation in level 3/4)
