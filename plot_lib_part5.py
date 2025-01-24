import numpy as np
import matplotlib.pyplot as plt

# Load data from CSV file, skipping the header
file_path = "lib_vs_lib_offload_results.csv"
data = np.genfromtxt(file_path, delimiter=",", dtype=str, encoding=None, skip_header=1)

# Extract relevant columns
methods = data[:, 0]  # "lib" or "lib_offload"
matrix_sizes = data[:, 1].astype(int)  # Matrix sizes
execution_times = data[:, 2].astype(float)  # Execution Time (s)
gflops = data[:, 3].astype(float)  # GFLOPS/s

# Fix Speedup Column: Strip spaces and replace "-" with NaN
speedup = np.array([float(x.strip()) if x.strip() != "-" else np.nan for x in data[:, 4]])

# Separate data for lib and lib_offload
lib_mask = methods == "lib"
lib_offload_mask = methods == "lib_offload"

matrix_sizes_lib = matrix_sizes[lib_mask]
gflops_lib = gflops[lib_mask]
matrix_sizes_offload = matrix_sizes[lib_offload_mask]
gflops_offload = gflops[lib_offload_mask]
speedup_offload = speedup[lib_offload_mask]

# Plot GFLOPS/s vs. Matrix Size
plt.figure(figsize=(10, 6))
plt.plot(matrix_sizes_lib, gflops_lib, marker="o", linestyle="-", label="lib (CPU)")
plt.plot(matrix_sizes_offload, gflops_offload, marker="s", linestyle="-", label="lib_offload (GPU)")
# plt.xscale("log")  # Log scale for matrix sizes
# plt.yscale("log")  # Log scale for GFLOPS/s comparison
plt.xlabel("Matrix Size")
plt.ylabel("GFLOPS/s")
plt.title("Performance (GFLOPS/s) vs. Matrix Size")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.savefig("gflops_vs_matrix_size_numpy.png")
plt.show()

# Plot Speedup vs. Matrix Size
plt.figure(figsize=(10, 6))
plt.plot(matrix_sizes_offload, speedup_offload, marker="o", linestyle="-", color="green", label="lib_offload Speedup")
# plt.xscale("log")  # Log scale for matrix sizes
plt.xlabel("Matrix Size")
plt.ylabel("Speedup (CPU time/GPU time)")
plt.title("Speedup vs. Matrix Size (lib_offload)")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.savefig("speedup_vs_matrix_size_numpy.png")
plt.show()
