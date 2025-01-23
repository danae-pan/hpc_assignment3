import numpy as np
import matplotlib.pyplot as plt

# Load data from CSV file
file_path = "team_size_performance_results.csv"

# Read the file, skipping the header
data = np.genfromtxt(file_path, delimiter=",", dtype=str, encoding=None, skip_header=1)

# Ensure data is properly structured
data = np.array([row.split(",") for row in data]) if data.ndim == 1 else data

# Extract relevant columns
methods = data[:, 0]  # Column 0: Method
teams = data[:, 2].astype(int)  # Column 2: Teams
kernel_times = data[:, 3].astype(float)  # Column 3: Kernel time
gflops = data[:, 4].astype(float)  # Column 4: GFLOPS/s

# Separate data for MKN and MNK methods
mkn_mask = methods == "mkn_offload"
mnk_mask = methods == "mnk_offload"

teams_mkn = teams[mkn_mask]
kernel_mkn = kernel_times[mkn_mask]
gflops_mkn = gflops[mkn_mask]

teams_mnk = teams[mnk_mask]
kernel_mnk = kernel_times[mnk_mask]
gflops_mnk = gflops[mnk_mask]

# Plot Kernel Execution Time vs. Team Size
plt.figure(figsize=(8, 5))
plt.plot(teams_mkn, kernel_mkn, marker='o', linestyle='-', label="MKN Offload")
plt.plot(teams_mnk, kernel_mnk, marker='s', linestyle='-', label="MNK Offload")

plt.xscale("log", base=2)
plt.xlabel("Number of Teams")
plt.ylabel("Kernel Execution Time (s)")
plt.title("Kernel Execution Time vs. Team Size")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.savefig("kernel_time_vs_teams.png")
plt.show()

# Plot GFLOPS/s vs. Team Size
plt.figure(figsize=(8, 5))
plt.plot(teams_mkn, gflops_mkn, marker='o', linestyle='-', label="MKN Offload")
plt.plot(teams_mnk, gflops_mnk, marker='s', linestyle='-', label="MNK Offload")

# plt.xscale("log", base=2)
plt.xlabel("Number of Teams")
plt.ylabel("GFLOPS/s")
plt.title("Performance (GFLOPS/s) vs. Team Size")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.savefig("gflops_vs_teams.png")
plt.show()
