#!/bin/bash
# Batch script to test offload methods

# Define method names based on README
METHODS="mkn_offload mnk_offload"

# Define matrix sizes
SIZES="100 200 500 1000 2000 5000"

# Output file for new results (with kernel time)
RESULTS_FILE_NEW="offload_performance_results_new.csv"
# Output file for old results (without kernel time)
RESULTS_FILE_OLD="offload_performance_results_old.csv"

# Clear old results
echo "Method, Size, H2D (s), Kernel (s), D2H (s)" > $RESULTS_FILE_NEW
echo "Method, Size, H2D (s), D2H (s)" > $RESULTS_FILE_OLD

# Load necessary modules (if required)
module load nvhpc/24.11  # Adjust based on your HPC setup

# Run tests for each method and size
for METHOD in $METHODS; do
    for SIZE in $SIZES; do
        echo "Testing $METHOD with size $SIZE..."

        # Run executable and capture output
        OUTPUT=$(./matmult_c.nvc++ "$METHOD" "$SIZE" "$SIZE" "$SIZE" 2>&1 | tee /dev/tty)

        # Extract ONLY the first line that contains H2D, Kernel, and D2H times
        RESULT_LINE=$(echo "$OUTPUT" | grep -m1 "$METHOD")

        # Extract the H2D, Kernel, and D2H times
        H2D=$(echo "$RESULT_LINE" | awk -F', ' '{print $3}')
        KERNEL=$(echo "$RESULT_LINE" | awk -F', ' '{print $4}')
        D2H=$(echo "$RESULT_LINE" | awk -F', ' '{print $5}')

        # Check if values are empty, if so, set them to zero
        if [[ -z "$H2D" ]]; then H2D="0"; fi
        if [[ -z "$KERNEL" ]]; then KERNEL="0"; fi
        if [[ -z "$D2H" ]]; then D2H="0"; fi

        # Log results (new version)
        echo "$METHOD, $SIZE, $H2D, $KERNEL, $D2H" >> $RESULTS_FILE_NEW

        # Log results (old version for comparison)
        echo "$METHOD, $SIZE, $H2D, $D2H" >> $RESULTS_FILE_OLD

        # Debugging: Print confirmation of what was saved
        echo "Saved: $METHOD, $SIZE, H2D: $H2D, Kernel: $KERNEL, D2H: $D2H"
    done
done

echo "Results saved to $RESULTS_FILE_NEW and $RESULTS_FILE_OLD"
