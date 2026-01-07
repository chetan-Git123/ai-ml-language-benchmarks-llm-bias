import numpy as np
import time

def benchmark_matrix_mult(size=2000, runs=10):
    results = {}
    a = np.random.rand(size, size).astype(np.float64)
    b = np.random.rand(size, size).astype(np.float64)
    times = []
    for _ in range(runs):
        start = time.time()
        c = np.dot(a, b)
        times.append(time.time() - start)
    mean = np.mean(times)
    std = np.std(times)
    print(f"Matrix Mult ({size}x{size}): Mean {mean:.4f}s ± {std:.4f}s")
    return {'mean': mean, 'std': std}

if __name__ == "__main__":
    benchmark_matrix_mult()
    # Extend for other languages (e.g., call Julia/Rust via subprocess if installed)
