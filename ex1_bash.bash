#!/bin/bash
# Profiling script for OpenMP and BLAS matrix multiplication implementations
#BSUB -J mm_batch_gpu
#BSUB -o mm_batch_gpu_%J.out
#BSUB -q hpcintrogpu
#BSUB -n 32
#BSUB -R "rusage[mem=2048]"
#BSUB -W 15
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"


# Define output file
RESULTS_FILE="performance_results.csv"

# Clear old results
echo "Method, Size, Execution Time (s), GFLOPS" > $RESULTS_FILE

# Define methods
METHODS="mkn_omp lib"

# Define matrix sizes
SIZES="100 200 500 1000 2000 5000"

# Load necessary modules (adjust if required)
module load nvhpc/24.11

# Run tests for each method and size
for METHOD in $METHODS; do
    for SIZE in $SIZES; do
        echo "Testing $METHOD with size $SIZE..."

        # Run executable and capture output
        OUTPUT=$(./matmult_c.nvc++ "$METHOD" "$SIZE" "$SIZE" "$SIZE" 2>&1 | tee /dev/tty)

        # Extract Execution Time and GFLOPS
        RESULT_LINE=$(echo "$OUTPUT" | grep -m1 "$METHOD")

        EXEC_TIME=$(echo "$RESULT_LINE" | awk -F', ' '{print $3}')
        GFLOPS=$(echo "$RESULT_LINE" | awk -F', ' '{print $4}')

        # Check if values are empty, if so, set them to zero
        EXEC_TIME=${EXEC_TIME:-0}
        GFLOPS=${GFLOPS:-0}

        # Log results
        echo "$METHOD, $SIZE, $EXEC_TIME, $GFLOPS" >> $RESULTS_FILE

        # Debugging: Print confirmation of what was saved
        echo "Saved: $METHOD, $SIZE, $EXEC_TIME, $GFLOPS"
    done
done

echo "Performance results saved to $RESULTS_FILE"
