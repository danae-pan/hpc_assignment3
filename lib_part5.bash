#!/bin/bash
# Batch script to compare CPU (lib) and GPU (lib_offload) DGEMM performance

#BSUB -J lib_vs_lib_offload
#BSUB -o lib_vs_lib_offload_%J.out
#BSUB -q hpcintrogpu
#BSUB -n 32
#BSUB -R "rusage[mem=2048]"
#BSUB -W 15
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"

# Define methods to test
METHODS="lib lib_offload"

# Define matrix sizes to experiment with
MATRIX_SIZES="100 500 1000 2000 5000 10000"

# Output file
RESULTS_FILE="lib_vs_lib_offload_results.csv"

# Clear old results
echo "Method, Matrix Size, Execution Time (s), GFLOPS/s, Speedup (CPU/GPU)" > $RESULTS_FILE

# Load necessary modules (adjust if required)
module load nvhpc/24.11

# Run tests for lib and lib_offload with different matrix sizes
for SIZE in $MATRIX_SIZES; do
    CPU_TIME=0  # Store CPU execution time for speedup calculation
    GPU_TIME=0  # Store GPU execution time for speedup calculation

    for METHOD in $METHODS; do
        echo "Testing $METHOD with matrix size $SIZE..."

        # Run executable and capture output
        OUTPUT=$(./matmult_c.nvc++ "$METHOD" "$SIZE" "$SIZE" "$SIZE")

        # Extract execution time
        RESULT_LINE=$(echo "$OUTPUT" | grep -m1 "$METHOD")
        EXEC_TIME=$(echo "$RESULT_LINE" | awk -F', ' '{print $3}')

        # Ensure non-empty execution time
        EXEC_TIME=${EXEC_TIME:-0}

        # Compute GFLOPS/s: (2 * m * n * k) / (Execution Time * 10^9)
        if [[ "$EXEC_TIME" == "0" || "$EXEC_TIME" == "" ]]; then
            GFLOPS="0"
        else
            GFLOPS=$(echo "scale=6; (2 * $SIZE * $SIZE * $SIZE) / ($EXEC_TIME * 10^9)" | bc)
        fi

        # Store execution times for speedup calculation
        if [[ "$METHOD" == "lib" ]]; then
            CPU_TIME=$EXEC_TIME
        elif [[ "$METHOD" == "lib_offload" ]]; then
            GPU_TIME=$EXEC_TIME
        fi

        # Log results
        echo "$METHOD, $SIZE, $EXEC_TIME, $GFLOPS, -" >> $RESULTS_FILE

        # Debugging: Print confirmation of what was saved
        echo "Saved: $METHOD, $SIZE, $EXEC_TIME, $GFLOPS"
    done

    # Compute speedup (CPU Time / GPU Time)
    if [[ "$GPU_TIME" != "0" && "$GPU_TIME" != "" ]]; then
        SPEEDUP=$(echo "scale=6; $CPU_TIME / $GPU_TIME" | bc)
        # Update results file with speedup
        sed -i "/lib_offload, $SIZE,/ s/-/$SPEEDUP/" $RESULTS_FILE
        echo "Speedup for size $SIZE: $SPEEDUP"
    fi
done

echo "Results saved to $RESULTS_FILE"
