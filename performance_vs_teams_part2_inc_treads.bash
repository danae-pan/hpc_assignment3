#!/bin/bash
# Batch script to test different team sizes and thread numbers for mkn_offload and mnk_offload

#BSUB -J problem2_inc_threads
#BSUB -o problem2_inc_threads_%J.out
#BSUB -q hpcintrogpu
#BSUB -n 32
#BSUB -R "rusage[mem=2048]"
#BSUB -W 15
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"

# Define methods to test
METHODS="mkn_offload mnk_offload"

# Define matrix size (fixed for this experiment)
MATRIX_SIZE=1000

# Define team sizes and thread limits to experiment with
TEAM_SIZES="64 128 256 512 1024 2048 4096"
THREAD_SIZES="32 64 128 256 512"

# Output file
RESULTS_FILE="team_thread_performance_results.csv"

# Clear old results
echo "Method, Matrix Size, Teams, Threads, Kernel (s), GFLOPS/s" > $RESULTS_FILE

# Load necessary modules (adjust if required)
module load nvhpc/24.11

# Run tests for MKN and MNK with different team and thread sizes
for TEAM in $TEAM_SIZES; do
    for THREAD in $THREAD_SIZES; do
        for METHOD in $METHODS; do
            echo "Testing $METHOD with team size $TEAM and thread limit $THREAD..."

            # Run executable with team and thread limit
            OUTPUT=$(./matmult_c.nvc++ "$METHOD" "$MATRIX_SIZE" "$MATRIX_SIZE" "$MATRIX_SIZE" "$TEAM" "$THREAD")

            # Extract Kernel time
            RESULT_LINE=$(echo "$OUTPUT" | grep -m1 "$METHOD")
            KERNEL=$(echo "$RESULT_LINE" | awk -F', ' '{print $5}')

            # Check if Kernel time is empty or zero, if so, set to zero
            KERNEL=${KERNEL:-0}

            # Compute GFLOPS/s: (2 * m * n * k) / (Kernel Time * 10^9)
            if [[ "$KERNEL" == "0" || "$KERNEL" == "" ]]; then
                GFLOPS="0"
            else
                GFLOPS=$(echo "scale=6; (2 * $MATRIX_SIZE * $MATRIX_SIZE * $MATRIX_SIZE) / ($KERNEL * 10^9)" | bc)
            fi

            # Log results
            echo "$METHOD, $MATRIX_SIZE, $TEAM, $THREAD, $KERNEL, $GFLOPS" >> $RESULTS_FILE

            # Debugging: Print confirmation of what was saved
            echo "Saved: $METHOD, $MATRIX_SIZE, $TEAM, $THREAD, $KERNEL, $GFLOPS"
        done
    done
done

echo "Results saved to $RESULTS_FILE"
