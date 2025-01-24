#!/bin/bash
# 02614 - High-Performance Computing, January 2024
# 
# batch script to run matmult on a dedicated GPU server in the hpcintrogpu
# queue
#
# Author: Bernd Dammann <bd@cc.dtu.dk>
#         Hans Henrik Brandenborg Sørensen <hhbs@dtu.dk>
#
#BSUB -J poisson_map_vs_memcpy
#BSUB -o poisson_map_vs_memcpy%J.out
#BSUB -q hpcintrogpu
#BSUB -n 32
#BSUB -R "rusage[mem=2048]"
#BSUB -W 00:15
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"

# Set OpenMP environment variables
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores

# Load NVIDIA HPC SDK
module load nvhpc/24.11

# Define output file
OUTPUT_FILE="poisson_results.data"

# Clear output file and add header
echo "Module Iterations GridSize ExecutionTime" > $OUTPUT_FILE

# Define grid sizes and methods
GRID_SIZES=(100 200 300 400 500)
METHODS=(0 1 2)  # 0 = OpenMP Map, 1 = OpenMP Target Memcpy, 2 = CPU

# Run experiments
for method in "${METHODS[@]}"; do
    for N in "${GRID_SIZES[@]}"; do
        echo "Running Poisson with N=$N, method=$method..."
        START_TIME=$(date +%s.%N)
        ITERATIONS=400  # Fixed number of iterations
        EXEC_TIME=$(./poisson $N $ITERATIONS $method | grep "Jacobi Offload Execution Time" | awk '{print $5}')
        END_TIME=$(date +%s.%N)
        
        # Save results to file
        echo "$method $ITERATIONS $N $EXEC_TIME" >> $OUTPUT_FILE
    done
done

echo "All experiments completed. Results saved to $OUTPUT_FILE"