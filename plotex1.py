import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "performance_results.csv"

# Read the CSV file while handling potential formatting issues
df = pd.read_csv(file_path)

# Display column names for debugging
print("Column names in CSV:", df.columns)

# Ensure column names are correctly stripped of any leading/trailing spaces
df.columns = df.columns.str.strip()

# Check the first few rows of the DataFrame
print(df.head())

# Ensure "Size" column exists and convert it to integer
if 'Size' not in df.columns:
    raise KeyError("Column 'Size' not found in CSV. Check column names:", df.columns)

# Ensure "GFLOPS" exists
if 'GFLOPS' not in df.columns:
    raise KeyError("Column 'GFLOPS' not found in CSV. Check column names:", df.columns)

# Filter for relevant methods
filtered_df = df[df['Method'].isin(['mkn_omp', 'lib'])]

# Pivot the DataFrame for easier plotting
pivot_df = filtered_df.pivot(index='Size', columns='Method', values='GFLOPS')

# Plot performance vs. matrix size
plt.figure(figsize=(8, 6))
for method in pivot_df.columns:
    plt.plot(pivot_df.index, pivot_df[method], marker='o', linestyle='-', label=method)

plt.xlabel("Matrix Size")
plt.ylabel("Performance (GFLOPS)")
plt.title("Performance vs. Matrix Size (mkn_omp vs lib)")
plt.legend()
plt.grid(True)

output_image_path = "performance_vs_size.png"
plt.savefig(output_image_path, dpi=300, bbox_inches="tight")

# Show the plot
plt.show()
