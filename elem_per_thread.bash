#!/bin/bash
# Batch script to vary ELEMENTS_PER_THREAD and test performance
#BSUB -J mm_batch_gpu
#BSUB -o mm_batch_gpu_%J.out
#BSUB -q hpcintrogpu
#BSUB -n 32
#BSUB -R "rusage[mem=2048]"
#BSUB -W 15
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#!/bin/bash
# Batch script to test different elements per thread for blk_offload
#!/bin/bash
# Batch script to test varying block sizes and elements per thread for blk_offload

METHOD="blk_offload"
SIZES="100 200 500 1000 2000 5000"
ELEMENTS="1 2 4 8"  # Number of elements computed per thread

RESULTS_FILE="offload_performance_blocks_vs_elements.csv"

echo "Method, Size, Elements Per Thread, Kernel (s), GFLOPS/s" > $RESULTS_FILE

module load nvhpc/24.11

for SIZE in $SIZES; do
        for ELEMENT in $ELEMENTS; do
            echo "Testing $METHOD with size $SIZE, elements per thread $ELEMENT..."

            # Run executable and capture output
            OUTPUT=$(./matmult_c.nvc++ "$METHOD" "$SIZE" "$SIZE" "$SIZE" "$ELEMENT" 2>&1 | tee /dev/tty)


            # Extract Kernel time
            RESULT_LINE=$(echo "$OUTPUT" | grep -m1 "$METHOD")

            KERNEL=$(echo "$RESULT_LINE" | awk -F', ' '{print $4}')

            # Compute GFLOPS/s
            if [[ "$KERNEL" == "0" || "$KERNEL" == "" ]]; then
                GFLOPS="0"
            else
                GFLOPS=$(echo "scale=6; (2 * $SIZE * $SIZE * $SIZE) / ($KERNEL * 10^9)" | bc)
            fi

            # Log results
            echo "$METHOD, $SIZE, $ELEMENT, $KERNEL, $GFLOPS" >> $RESULTS_FILE
            echo "Saved: $METHOD, $SIZE, $ELEMENT, $KERNEL, $GFLOPS"
        done
done

echo "Results saved to $RESULTS_FILE"
