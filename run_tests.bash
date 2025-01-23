!/bin/bash
# Batch script to test offload methods with kernel time logging

#BSUB -J mm_batch_gpu
#BSUB -o mm_batch_gpu_%J.out
#BSUB -q hpcintrogpu
#BSUB -n 32
#BSUB -R "rusage[mem=2048]"
#BSUB -W 15
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"

# Define method names based on README
METHODS="mkn_offload mnk_offload blk_offload asy_offload"

# Define matrix sizes
SIZES="100 200 500 1000 2000 5000"

# Output file
RESULTS_FILE="offload_performance_resultsnumber2real.csv"

# Clear old results and write header
echo "Method, Size, H2D (s), Kernel (s), D2H (s)" > "$RESULTS_FILE"

# Load necessary modules
module load nvhpc/24.11  # Adjust if needed for your HPC setup

# Run tests for each method and size
for METHOD in $METHODS; do
    for SIZE in $SIZES; do
        echo "Testing $METHOD with size $SIZE..."

        # Run executable and capture output
        OUTPUT=$(./matmult_c.nvc++ "$METHOD" "$SIZE" "$SIZE" "$SIZE" 2>&1 | tee /dev/tty)

        # Extract performance metrics
        RESULT_LINE=$(echo "$OUTPUT" | grep -m1 "$METHOD")

        # Extract H2D, Kernel, and D2H times
        H2D=$(echo "$RESULT_LINE" | awk -F', ' '{print $3}')
        KERNEL=$(echo "$RESULT_LINE" | awk -F', ' '{print $4}')
        D2H=$(echo "$RESULT_LINE" | awk -F', ' '{print $5}')

        # Default to 0 if values are missing
        H2D=${H2D:-0}
        KERNEL=${KERNEL:-0}
        D2H=${D2H:-0}

        # Log results
        echo "$METHOD, $SIZE, $H2D, $KERNEL, $D2H" >> "$RESULTS_FILE"

        # Debugging: Print confirmation of what was saved
        echo "Saved: $METHOD, $SIZE, H2D: $H2D, Kernel: $KERNEL, D2H: $D2H"
    done
done

echo "Results saved to $RESULTS_FILE"
