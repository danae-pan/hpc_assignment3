#!/bin/bash
# 02614 - High-Performance Computing, January 2024
# Batch script to run Poisson solver on a dedicated GPU server in the hpcintrogpu queue

#BSUB -J poisson_speedup
#BSUB -o poisson_speedup_%J.out
#BSUB -q hpcintrogpu
#BSUB -n 32
#BSUB -R "rusage[mem=2048]"
#BSUB -W 00:15
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=2:mode=exclusive_process"  # Request 2 GPUs for comparison

# Set OpenMP environment variables
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
export OMP_NUM_DEVICES=2
export OMP_TARGET_OFFLOAD=MANDATORY
export CUDA_VISIBLE_DEVICES=0,1  # Ensure both GPUs are visible

# Load NVIDIA HPC SDK
module load nvhpc/24.11

# Define output file
SPEEDUP_FILE="poisson_speedup_analysis.data"

# Clear output file and add header
echo "GridSize Iterations ExecTime_1GPU ExecTime_2GPU Speedup" > $SPEEDUP_FILE

# Define grid sizes and iteration counts
GRID_SIZES=(64 128 256 384)
ITERATION_COUNTS=(100 200 400 800)  # Different iteration values

# Run experiments
for N in "${GRID_SIZES[@]}"; do
    for ITER in "${ITERATION_COUNTS[@]}"; do
        echo "Running Poisson solver for GridSize=$N, Iterations=$ITER..."

        # Run with 1 GPU
        EXEC_TIME_1GPU=$(./poisson $N $ITER 1 2| grep "Jacobi Offload Execution Time" | awk '{print $5}')

        # Run with 2 GPUs
        EXEC_TIME_2GPU=$(./poisson $N $ITER 2 | grep "Jacobi Offload Execution Time" | awk '{print $5}')

        # Calculate speed-up
        SPEEDUP=$(echo "scale=6; $EXEC_TIME_1GPU / $EXEC_TIME_2GPU" | bc)

        # Save results
        echo "$N $ITER $EXEC_TIME_1GPU $EXEC_TIME_2GPU $SPEEDUP" >> $SPEEDUP_FILE
    done
done

echo "All experiments completed. Speed-up results saved to $SPEEDUP_FILE"