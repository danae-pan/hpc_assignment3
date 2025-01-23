import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data with delimiter enforcement and strip spaces from headers
file_path = "offload_performance_resultsnumber2real.csv"
df = pd.read_csv(file_path, delimiter=",")

# Debugging: Print column names to check if they match expected ones
print("Column names found in CSV:", df.columns.tolist())

# Strip any leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Ensure correct column names by renaming if necessary
df.rename(columns={
    "Method": "Method",
    "Size": "Size",
    "H2D (s)": "H2D (s)",
    "Kernel (s)": "Kernel (s)",
    "D2H (s)": "D2H (s)"
}, inplace=True)

# Compute total execution time and data transfer overhead
df["Total Time"] = df["H2D (s)"] + df["Kernel (s)"] + df["D2H (s)"]
df["Transfer Overhead (%)"] = (df["H2D (s)"] + df["D2H (s)"]) / df["Total Time"] * 100

# Unique methods and sizes
methods = df["Method"].unique()
sizes = sorted(df["Size"].unique())

# --- Plot 1: Stacked Bar Chart for Time Breakdown ---
plt.figure(figsize=(10, 6))
for method in methods:
    subset = df[df["Method"] == method]
    plt.bar(subset["Size"], subset["H2D (s)"], label=f"{method} H2D", alpha=0.7)
    plt.bar(subset["Size"], subset["Kernel (s)"], bottom=subset["H2D (s)"], label=f"{method} Kernel", alpha=0.7)
    plt.bar(subset["Size"], subset["D2H (s)"], bottom=subset["H2D (s)"] + subset["Kernel (s)"], label=f"{method} D2H", alpha=0.7)

plt.xlabel("Matrix Size")
plt.ylabel("Time (s)")
plt.legend()
plt.title("Breakdown of Execution Time (H2D, Kernel, D2H)")
plt.xticks(sizes, labels=[str(size) for size in sizes], rotation=45)
plt.savefig("time_breakdown.png")
plt.show()

# --- Plot 2: Transfer Overhead (%) ---
plt.figure(figsize=(10, 6))
for method in methods:
    subset = df[df["Method"] == method]
    plt.plot(subset["Size"], subset["Transfer Overhead (%)"], marker="o", label=method)

plt.xlabel("Matrix Size")
plt.ylabel("Data Transfer Overhead (%)")
plt.legend()
plt.title("Data Transfer Overhead vs. Matrix Size")
plt.xticks(sizes, labels=[str(size) for size in sizes], rotation=45)
plt.savefig("transfer_overhead.png")
plt.show()

# --- Plot 3: GFLOPS/s vs. Matrix Size ---
df["GFLOPS/s"] = (2 * df["Size"] ** 3) / (df["Kernel (s)"] * 1e9)

plt.figure(figsize=(10, 6))
for method in methods:
    subset = df[df["Method"] == method]
    plt.plot(subset["Size"], subset["GFLOPS/s"], marker="o", label=method)

plt.xlabel("Matrix Size")
plt.ylabel("Performance (GFLOPS/s)")
plt.legend()
plt.title("Performance Comparison (GFLOPS/s)")
plt.xticks(sizes, labels=[str(size) for size in sizes], rotation=45)
plt.savefig("performance_comparison.png")
plt.show()

# --- Plot 4: Transfer Overhead vs. GFLOPS ---
plt.figure(figsize=(10, 6))
for method in methods:
    subset = df[df["Method"] == method]
    plt.scatter(subset["Transfer Overhead (%)"], subset["GFLOPS/s"], label=method)

plt.xlabel("Data Transfer Overhead (%)")
plt.ylabel("Performance (GFLOPS/s)")
plt.legend()
plt.title("Data Transfer Impact on Performance")
plt.xticks(sizes, labels=[str(size) for size in sizes], rotation=45)
plt.savefig("transfer_vs_performance.png")
plt.show()
