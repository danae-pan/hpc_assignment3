#!/bin/bash
# Batch script to test all offload methods and compute GFLOPS/s
# Batch script for profiling all matrix multiplication implementations using NSight Systems (NSYS) and Nsight Compute (NCU)

#BSUB -J mm_batch_gpu
#BSUB -o mm_batch_gpu_%J.out
#BSUB -q hpcintrogpu
#BSUB -n 32
#BSUB -R "rusage[mem=2048]"
#BSUB -W 15
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"

# Define method names
METHODS="mkn_offload mnk_offload blk_offload"

# Define matrix sizes
SIZES="100 200 500 1000 2000 5000"

# Output file
RESULTS_FILE="offload_performance_noasysnum5.csv"

# Clear old results
echo "Method, Size, Kernel (s), GFLOPS/s" > $RESULTS_FILE

# Load necessary modules (adjust if required)
module load nvhpc/24.11

# Run tests for each method and size
for METHOD in $METHODS; do
    for SIZE in $SIZES; do
        echo "Testing $METHOD with size $SIZE..."

        # Run executable and capture output
        OUTPUT=$(./matmult_c.nvc++ "$METHOD" "$SIZE" "$SIZE" "$SIZE" 2>&1 | tee /dev/tty)

        # Extract Kernel time
        RESULT_LINE=$(echo "$OUTPUT" | grep -m1 "$METHOD")

        KERNEL=$(echo "$RESULT_LINE" | awk -F', ' '{print $4}')

        # Check if Kernel time is empty or zero, if so, set to zero
        KERNEL=${KERNEL:-0}

        # Compute GFLOPS/s: (2 * m * n * k) / (Kernel Time * 10^9)
        if [[ "$KERNEL" == "0" || "$KERNEL" == "" ]]; then
            GFLOPS="0"
        else
            GFLOPS=$(echo "scale=6; (2 * $SIZE * $SIZE * $SIZE) / ($KERNEL * 10^9)" | bc)
        fi

        # Log results
        echo "$METHOD, $SIZE, $KERNEL, $GFLOPS" >> $RESULTS_FILE

        # Debugging: Print confirmation of what was saved
        echo "Saved: $METHOD, $SIZE, $KERNEL, $GFLOPS"
    done
done

echo "Results saved to $RESULTS_FILE"
