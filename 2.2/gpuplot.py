import pandas as pd
import matplotlib.pyplot as plt

# Define input data file for GPU
data_file = "jacobi_mlups_gpu.data"

# Load the data into a Pandas DataFrame
df = pd.read_csv(data_file, sep='\s+')

# Extract unique iteration counts and methods
iterations = sorted(df["Iterations"].unique())
methods = sorted(df["Method"].unique())  # 3 = Plain, 4 = Norm-based
matrix_sizes = sorted(df["GridSize"].unique())

# Define markers and line styles for different methods
marker_styles = {1: "o", 4: "s"}  # Circle for plain, square for norm-based
line_styles = {1: "--", 4: "-"}   # Dashed for plain, solid for norm-based
colors = ["b", "r", "g", "m", "c", "y"]  # Different colors for different iterations

plt.figure(figsize=(12, 7))

# Plot MLUPS for GPU
for method in methods:
    if method not in marker_styles:  # Handle unexpected method values
        print(f"Skipping unknown method: {method}")
        continue

    for i, iter_count in enumerate(iterations):
        subset = df[(df["Iterations"] == iter_count) & (df["Method"] == method)]
        if subset.empty:
            continue  # Skip if no data for this method & iteration

        plt.plot(subset["GridSize"], subset["MLUPS_GPU"], 
                 marker=marker_styles[method], linestyle=line_styles[method], 
                 label=f"GPU - {iter_count} Iter (Method {method})", color=colors[i])
        
# Labels and title
plt.xlabel("Matrix Size (N)")
plt.ylabel("MLUPS (Millions of Lattice Updates per Second)")
plt.title("Jacobi Solver MLUPS Performance - GPU Only")
plt.legend()
plt.grid(True)

# Save and show plot
plt.savefig("mlups_performance_gpu.png", dpi=300)
plt.show()
