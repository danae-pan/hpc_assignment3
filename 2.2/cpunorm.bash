#!/bin/bash
# 02614 - High-Performance Computing, January 2024
# Batch script to run Jacobi solver on CPU

#BSUB -J jacobi_mlups_cpu
#BSUB -o jacobi_mlups_cpu_%J.out
#BSUB -q hpcintrogpu
#BSUB -n 32
#BSUB -R "rusage[mem=2048]"
#BSUB -W 00:15
#BSUB -R "span[hosts=1]"

# Load NVIDIA HPC SDK
module load nvhpc/24.11

# Define output file
RESULT_FILE="jacobi_mlups_cpu.data"

# Clear output file and add header
echo "GridSize Iterations Method ExecTime_CPU MLUPS_CPU" > $RESULT_FILE

# Define grid sizes and iteration counts
GRID_SIZES=(64 128 256 384)
ITERATION_COUNTS=(100 200 400 800)
METHODS=(3 4)  # 3 = Plain Jacobi, 4 = Norm-based Jacobi
TOLERANCE=1e-1

# Path to the executable
EXECUTABLE="./poisson"

# Run CPU experiments
for N in "${GRID_SIZES[@]}"; do
    for ITER in "${ITERATION_COUNTS[@]}"; do
        for METHOD in "${METHODS[@]}"; do
            echo "Running CPU Jacobi solver for GridSize=$N, Iterations=$ITER, Method=$METHOD..."

            # Run on CPU (offload=0)
            START_TIME_CPU=$(date +%s.%N)
            EXEC_TIME_CPU=$($EXECUTABLE $N $ITER 0 $METHOD $TOLERANCE | grep "Jacobi Offload Execution Time" | awk '{print $5}')
            END_TIME_CPU=$(date +%s.%N)
            TOTAL_TIME_CPU=$(echo "$END_TIME_CPU - $START_TIME_CPU" | bc)

            # Calculate MLUPS
            GRID_POINTS=$(echo "$N * $N * $N" | bc)
            MLUPS_CPU=$(echo "scale=6; $GRID_POINTS * $ITER / ($TOTAL_TIME_CPU * 10^6)" | bc)

            # Save results
            echo "$N $ITER $METHOD $TOTAL_TIME_CPU $MLUPS_CPU" >> $RESULT_FILE
        done
    done
done

echo "CPU experiments completed. Results saved to $RESULT_FILE."
