import pandas as pd
import matplotlib.pyplot as plt

# Define input data file for CPU
data_file = "jacobi_mlups_cpu.data"

# Load the data into a Pandas DataFrame
df = pd.read_csv(data_file, delim_whitespace=True)

# Extract unique iteration counts and methods
iterations = sorted(df["Iterations"].unique())
methods = sorted(df["Method"].unique())  # 3 = Plain, 4 = Norm-based
matrix_sizes = sorted(df["GridSize"].unique())

# Define markers and line styles for different methods
marker_styles = {3: "o", 4: "s"}  # Circle for plain, square for norm-based
line_styles = {3: "--", 4: "-"}   # Dashed for plain, solid for norm-based
colors = ["b", "r", "g", "m", "c", "y"]  # Different colors for different iterations

plt.figure(figsize=(12, 7))

# Plot MLUPS for CPU
for method in methods:
    for i, iter_count in enumerate(iterations):
        subset = df[(df["Iterations"] == iter_count) & (df["Method"] == method)]
        plt.plot(subset["GridSize"], subset["MLUPS_CPU"], 
                 marker=marker_styles[method], linestyle=line_styles[method], 
                 label=f"CPU - {iter_count} Iter (Method {method})", color=colors[i])

# Labels and title
plt.xlabel("Matrix Size (N)")
plt.ylabel("MLUPS (Millions of Lattice Updates per Second)")
plt.title("Jacobi Solver MLUPS Performance - CPU Only")
plt.legend()
plt.grid(True)

# Save and show plot
plt.savefig("mlups_performance_cpu.png", dpi=300)
plt.show()
