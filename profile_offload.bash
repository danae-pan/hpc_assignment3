#!/bin/bash
# Batch script for profiling all matrix multiplication implementations using NSight Systems (NSYS) and Nsight Compute (NCU)

#BSUB -J mm_batch_gpu
#BSUB -o mm_batch_gpu_%J.out
#BSUB -q hpcintrogpu
#BSUB -n 32
#BSUB -R "rusage[mem=2048]"
#BSUB -W 15
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"

# Define output directory for logs
OUTPUT_DIR="profiling_results"
mkdir -p $OUTPUT_DIR

# Define methods and sizes
METHODS="mkn_offload mnk_offload asy_offload"
SIZES="100 200 500 1000 2000 5000"

# Load necessary module
module load nvhpc/24.11

# Set profiling environment variables
export TMPDIR=$__LSF_JOB_TMPDIR__
export MFLOPS_MAX_IT=1
export MATMULT_COMPARE=0

# Loop over all methods and sizes
for METHOD in $METHODS; do
    for SIZE in $SIZES; do
        echo "Profiling $METHOD with matrix size $SIZE..."

        # Define unique file names
        NSYS_FILE="$OUTPUT_DIR/nsys_${METHOD}_${SIZE}"
        NCU_FILE="$OUTPUT_DIR/ncu_${METHOD}_${SIZE}"

        # Run NSYS profiling (execution timeline)
        echo "Starting NSYS profiling for $METHOD, Size $SIZE"
        nsys profile -o "$NSYS_FILE" \
            --stats=true \
            ./matmult_c.nvc++ $METHOD $SIZE $SIZE $SIZE 
        
        echo "NSYS profiling completed for $METHOD, Size $SIZE"

        # Run NCU profiling (memory and compute performance)
        echo "Starting NCU profiling for $METHOD, Size $SIZE"
        ncu -o "$NCU_FILE" \
            --set basic \
            --section MemoryWorkloadAnalysis \
            --section MemoryWorkloadAnalysis_Chart \
            --section ComputeWorkloadAnalysis \
            ./matmult_c.nvc++ $METHOD $SIZE $SIZE $SIZE 

        echo "NCU profiling completed for $METHOD, Size $SIZE"
    done
done

echo "Profiling complete! Results saved in $OUTPUT_DIR"
