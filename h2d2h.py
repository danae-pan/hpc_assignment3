import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
file_path = "offload_performance_resultsnumber2real.csv"
df = pd.read_csv(file_path, delimiter=",")

# Strip any leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Compute total execution time and data transfer overhead
df["Total Time"] = df["H2D (s)"] + df["Kernel (s)"] + df["D2H (s)"]
df["Transfer Overhead (%)"] = (df["H2D (s)"] + df["D2H (s)"]) / df["Total Time"] * 100

# Unique methods and sizes
methods = df["Method"].unique()
sizes = sorted(df["Size"].unique())

# --- Scatter Plot for Execution Time Breakdown ---
plt.figure(figsize=(10, 6))
for method in methods:
    subset = df[df["Method"] == method]
    plt.scatter(subset["Size"], subset["H2D (s)"], label=f"{method} H2D", marker="o", alpha=0.7)
    plt.scatter(subset["Size"], subset["Kernel (s)"], label=f"{method} Kernel", marker="x", alpha=0.7)
    plt.scatter(subset["Size"], subset["D2H (s)"], label=f"{method} D2H", marker="s", alpha=0.7)

plt.xlabel("Matrix Size")
plt.ylabel("Time (s)")
plt.legend()
plt.title("Execution Time Breakdown (H2D, Kernel, D2H)")
plt.xticks(sizes, labels=[str(size) for size in sizes], rotation=45)  # Rotate for better visibility
plt.yscale("log")  # Keep log scale for better visibility
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.savefig("scatter_time_breakdown.png")
plt.show()

# --- H2D and D2H Transfer Times ---
plt.figure(figsize=(10, 6))
for method in methods:
    subset = df[df["Method"] == method]
    plt.plot(subset["Size"], subset["H2D (s)"], marker="o", linestyle="-", label=f"{method} H2D")
    plt.plot(subset["Size"], subset["D2H (s)"], marker="s", linestyle="--", label=f"{method} D2H")

plt.xlabel("Matrix Size")
plt.ylabel("Transfer Time (seconds)")
plt.title("H2D and D2H Transfer Times for Offload Methods")
plt.xticks(sizes, labels=[str(size) for size in sizes], rotation=45)
plt.yscale("log")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.savefig("transfer_times.png")
plt.show()
