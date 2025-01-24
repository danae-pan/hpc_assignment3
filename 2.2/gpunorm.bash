#!/bin/bash
# 02614 - High-Performance Computing, January 2024
# Batch script to run Jacobi solver on GPU

#BSUB -J jacobi_mlups_gpu
#BSUB -o jacobi_mlups_gpu_%J.out
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
RESULT_FILE="jacobi_mlups_gpu.data"

# Clear output file and add header
echo "GridSize Iterations Method ExecTime_GPU MLUPS_GPU" > $RESULT_FILE

# Define grid sizes and iteration counts
GRID_SIZES=(64 128 256 384)
ITERATION_COUNTS=(100 200 400 800)
METHODS=(1 4)  # 1 = Plain Jacobi, 4 = Norm-based Jacobi
TOLERANCE=1e-1

# Path to the executable
EXECUTABLE="./poisson"

# Run GPU experiments
for N in "${GRID_SIZES[@]}"; do
    for ITER in "${ITERATION_COUNTS[@]}"; do
        for METHOD in "${METHODS[@]}"; do
            echo "Running GPU Jacobi solver for GridSize=$N, Iterations=$ITER, Method=$METHOD..."

            # Run on GPU 1 (offload=1)
            START_TIME_GPU=$(date +%s.%N)
            EXEC_TIME_GPU=$($EXECUTABLE $N $ITER 1 $METHOD $TOLERANCE | grep "Jacobi Offload Execution Time" | awk '{print $5}')
            END_TIME_GPU=$(date +%s.%N)
            TOTAL_TIME_GPU=$(echo "$END_TIME_GPU - $START_TIME_GPU" | bc)

            # Calculate MLUPS
            GRID_POINTS=$(echo "$N * $N * $N" | bc)
            MLUPS_GPU=$(echo "scale=6; $GRID_POINTS * $ITER / ($TOTAL_TIME_GPU * 10^6)" | bc)

            # Save results
            echo "$N $ITER $METHOD $TOTAL_TIME_GPU $MLUPS_GPU" >> $RESULT_FILE
        done
    done
done

echo "GPU experiments completed. Results saved to $RESULT_FILE."
