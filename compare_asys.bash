#!/bin/bash
# Batch script to test all offload methods

# Define method names
METHODS="mkn_offload mnk_offload asy_offload"

# Define matrix sizes
SIZES="100 200 500 1000 2000 5000"

# Output file
RESULTS_FILE="offload_performance_results.csv"

# Clear old results
echo "Method, Size, H2D (s), Kernel (s), D2H (s), CMR" > $RESULTS_FILE

# Load necessary modules (adjust if required)
module load nvhpc/24.11

# Run tests for each method and size
for METHOD in $METHODS; do
    for SIZE in $SIZES; do
        echo "Testing $METHOD with size $SIZE..."

        # Run executable and capture output
        OUTPUT=$(./matmult_c.nvc++ "$METHOD" "$SIZE" "$SIZE" "$SIZE" 2>&1 | tee /dev/tty)

        # Extract H2D, Kernel, and D2H times
        RESULT_LINE=$(echo "$OUTPUT" | grep -m1 "$METHOD")

        H2D=$(echo "$RESULT_LINE" | awk -F', ' '{print $3}')
        KERNEL=$(echo "$RESULT_LINE" | awk -F', ' '{print $4}')
        D2H=$(echo "$RESULT_LINE" | awk -F', ' '{print $5}')

        # Check if values are empty, if so, set them to zero
        H2D=${H2D:-0}
        KERNEL=${KERNEL:-0}
        D2H=${D2H:-0}

        # Compute CMR (Compute-to-Memory Ratio)
        CMR=$(echo "$KERNEL / ($H2D + $D2H)" | bc -l 2>/dev/null || echo "0")

        # Log results
        echo "$METHOD, $SIZE, $H2D, $KERNEL, $D2H, $CMR" >> $RESULTS_FILE

        # Debugging: Print confirmation of what was saved
        echo "Saved: $METHOD, $SIZE, $H2D, $KERNEL, $D2H, $CMR"
    done
done

echo "Results saved to $RESULTS_FILE"
