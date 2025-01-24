import pandas as pd
import matplotlib.pyplot as plt

# Define input data file
data_file = "jacobi_mlups_performance.data"

# Load the data into a Pandas DataFrame
df = pd.read_csv(data_file, delim_whitespace=True)

# Extract unique iteration counts
iterations = sorted(df["Iterations"].unique())
matrix_sizes = sorted(df["GridSize"].unique())

# Plot MLUPS for CPU and GPU
plt.figure(figsize=(10, 6))

for iter_count in iterations:
    subset = df[df["Iterations"] == iter_count]
    plt.plot(subset["GridSize"], subset["MLUPS_CPU"], marker="o", linestyle="--", label=f"CPU - {iter_count} Iter")
    plt.plot(subset["GridSize"], subset["MLUPS_GPU"], marker="s", linestyle="-", label=f"GPU - {iter_count} Iter")

# Labels and title
plt.xlabel("Matrix Size (N)")
plt.ylabel("MLUPS (Millions of Lattice Updates per Second)")
plt.title("Jacobi Solver MLUPS Performance (CPU vs GPU)")
plt.legend()
plt.grid(True)

# Save and show plot
plt.savefig("mlups_performance.png", dpi=300)
plt.show()
