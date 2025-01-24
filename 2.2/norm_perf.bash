#!/bin/bash
# 02614 - High-Performance Computing, January 2024
# Batch script to run Jacobi solver on a dedicated GPU server in the hpcintrogpu queue

#BSUB -J jacobi_mlups_performance
#BSUB -o jacobi_mlups_performance_%J.out
#BSUB -q hpcintrogpu
#BSUB -n 32
#BSUB -R "rusage[mem=2048]"
#BSUB -W 00:15
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU

# Set OpenMP environment variables
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
export OMP_NUM_DEVICES=1
export OMP_TARGET_OFFLOAD=MANDATORY
export CUDA_VISIBLE_DEVICES=0  # Ensure the GPU is visible

# Load NVIDIA HPC SDK
module load nvhpc/24.11

# Define output file
RESULT_FILE="jacobi_mlups_performance.data"

# Clear output file and add header
echo "GridSize Iterations ExecTime_CPU ExecTime_GPU MLUPS_CPU MLUPS_GPU" > $RESULT_FILE

# Define grid sizes and iteration counts
GRID_SIZES=(64 128 256 384)
ITERATION_COUNTS=(100 200 400 800)  # Different iteration values
TOLERANCE=1e-3  # Convergence tolerance

# Path to the executable
EXECUTABLE="./poisson"

# Run experiments
for N in "${GRID_SIZES[@]}"; do
    for ITER in "${ITERATION_COUNTS[@]}"; do
        echo "Running Jacobi norm-based solver for GridSize=$N, Iterations=$ITER..."

        # Run on CPU (offload=0, method=4)
        START_TIME_CPU=$(date +%s.%N)
        $EXECUTABLE $N $ITER 0 4 $TOLERANCE | grep "Jacobi Offload Execution Time"
        END_TIME_CPU=$(date +%s.%N)
        EXEC_TIME_CPU=$(echo "$END_TIME_CPU - $START_TIME_CPU" | bc)

        # Run on GPU 1 (offload=1, method=4)
        START_TIME_GPU=$(date +%s.%N)
        $EXECUTABLE $N $ITER 1 4 $TOLERANCE | grep "Jacobi Offload Execution Time"
        END_TIME_GPU=$(date +%s.%N)
        EXEC_TIME_GPU=$(echo "$END_TIME_GPU - $START_TIME_GPU" | bc)

        # Calculate MLUPS
        GRID_POINTS=$(echo "$N * $N * $N" | bc)
        MLUPS_CPU=$(echo "scale=6; $GRID_POINTS * $ITER / ($EXEC_TIME_CPU * 10^6)" | bc)
        MLUPS_GPU=$(echo "scale=6; $GRID_POINTS * $ITER / ($EXEC_TIME_GPU * 10^6)" | bc)

        # Save results
        echo "$N $ITER $EXEC_TIME_CPU $EXEC_TIME_GPU $MLUPS_CPU $MLUPS_GPU" >> $RESULT_FILE
    done
done

echo "All experiments completed. Results saved to $RESULT_FILE."
