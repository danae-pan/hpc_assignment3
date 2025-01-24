import pandas as pd
import matplotlib.pyplot as plt

# Load data and clean column names
file_path = "offload_performance_noasys.csv"
df = pd.read_csv(file_path)

# Debug: Print actual column names to verify issues
print("Columns in CSV:", df.columns.tolist())

# Strip spaces and standardize column names
df.columns = df.columns.str.strip()

# Convert 'Size' and 'GFLOPS/s' to numeric
df["Size"] = pd.to_numeric(df["Size"], errors="coerce")
df["GFLOPS/s"] = pd.to_numeric(df["GFLOPS/s"], errors="coerce")

# Drop rows where 'Size' or 'GFLOPS/s' could not be converted
df = df.dropna(subset=["Size", "GFLOPS/s"])

# Unique methods
methods = df["Method"].unique()

# --- Plot: GFLOPS/s vs. Matrix Size ---
plt.figure(figsize=(10, 6))

for method in methods:
    subset = df[df["Method"] == method]
    plt.plot(subset["Size"], subset["GFLOPS/s"], marker="o", linestyle="-", label=method)

plt.xlabel("Matrix Size")
plt.ylabel("Performance (GFLOPS/s)")
plt.legend()
plt.title("Performance Comparison of Offload Methods")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.savefig("offload_performance_plot.png")
plt.show()
