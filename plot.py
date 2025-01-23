import pandas as pd
import matplotlib.pyplot as plt

# Load the performance results from the CSV file
csv_file = "offload_performance_results_new.csv"  # Ensure this file exists
df = pd.read_csv(csv_file)

# Debug: Print the column names to check for unexpected spaces or issues
print("Column Names in CSV:", df.columns.tolist())

# Strip any unexpected spaces in column names
df.columns = df.columns.str.strip()

# Ensure 'Size' column exists after stripping spaces
if "Size" not in df.columns:
    raise KeyError("The 'Size' column is missing. Check the CSV formatting.")

# Convert 'Size' column to integer for proper plotting
df["Size"] = df["Size"].astype(int)

# Create the plot
plt.figure(figsize=(10, 6))

for method in df["Method"].unique():
    subset = df[df["Method"] == method]
    plt.plot(subset["Size"], subset["H2D (s)"], marker='o', linestyle='-', label=f"{method} H2D")
    plt.plot(subset["Size"], subset["D2H (s)"], marker='s', linestyle='--', label=f"{method} D2H")
    plt.plot(subset["Size"], subset["Kernel (s)"], marker='^', linestyle='-.', label=f"{method} Kernel")

# Labels and title
plt.xlabel("Matrix Size")
plt.ylabel("Time (seconds)")
plt.title("H2D, D2H, and Kernel Execution Times for Offload Methods")
plt.legend()
plt.grid(True)

# Save the plot as an image
plt.savefig("offload_performance_plot_new.png")

# Show the plot
plt.show()
