#include <stdio.h>
#include <omp.h>



extern "C" {
void matmult_mkn_omp(int m, int n, int k, double **A, double **B, double **C) {
    double start = omp_get_wtime();


    #pragma omp parallel shared(A,B,C)
    {
        #pragma omp for
        for(int i=0;i<m;i++){
            for(int l=0;l<k;l++){
                for(int j=0;j<n;j++){
                    // #pragma omp atomic
                    C[i][j] += A[i][l]*B[l][j];
                }
            }
        }
        
        double end = omp_get_wtime();
        double exec_time = end - start;

        // Compute GFLOPS: 2 * m * n * k / (time * 10^9)
        double gflops = (2.0 * m * n * k) / (exec_time * 1.0e9);

        // Print results in the same format
        printf("mkn_omp, %d, %f, %f\n", m, exec_time, gflops);
    }
    }
}





extern "C" {
    #include <cblas.h>
    void matmult_lib(int m, int n, int k, double **A, double **B, double **C) {
        double start = omp_get_wtime();

        // Call BLAS function
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    m, n, k, 1.0, *A, k, *B, n, 0.0, *C, n);

        double end = omp_get_wtime();
        double exec_time = end - start;

        // Compute GFLOPS
        double gflops = (2.0 * m * n * k) / (exec_time * 1.0e9);

        // Print results in the same format
        printf("matmult_lib, %d, %f, %f\n", m, exec_time, gflops);
    }
}

// Function to initialize data transfers to GPU (H2D)
void initialize_offload(int m, int n, int k, double **A, double **B, double **C, double *h2d_time) {
    double start_H2D = omp_get_wtime();
    
    // Transfer data to device
    #pragma omp target enter data map(to: A[0:m][0:k], B[0:k][0:n]) map(alloc: C[0:m][0:n])

    *h2d_time = omp_get_wtime() - start_H2D;  // Store H2D time
}

// Function to finalize and retrieve results from GPU (D2H)
void finalize_offload(int m, int n, int k, double **A, double **B, double **C, double *d2h_time) {
    double start_D2H = omp_get_wtime();
    
    // Transfer results back to host
    #pragma omp target exit data map(from: C[0:m][0:n])

    *d2h_time = omp_get_wtime() - start_D2H;  // Store D2H time
}

// OpenMP MKN Offload
extern "C" {
void matmult_mkn_offload(int m, int n, int k, double **A, double **B, double **C) {
    double h2d_time, d2h_time, kernel_time;  // Timing variables

    initialize_offload(m, n, k, A, B, C, &h2d_time);  // Start GPU offload

    double start_kernel = omp_get_wtime();

    // Perform matrix multiplication on GPU
    #pragma omp target teams distribute parallel for collapse(2) num_teams(min(256, m)) thread_limit(128)
    for (int i = 0; i < m; i++) {
        for (int l = 0; l < k; l++) {
            for (int j = 0; j < n; j++) {
                C[i][j] += A[i][l] * B[l][j];  // Matrix multiplication
            }
        }
    }

    kernel_time = omp_get_wtime() - start_kernel;

    finalize_offload(m, n, k, A, B, C, &d2h_time);  // Retrieve results from GPU

    // Ensure only master thread prints results
    #pragma omp master
    {
        #pragma omp flush
        printf("mkn_offload, %d, %f, %f, %f\n", m, h2d_time, kernel_time, d2h_time);
    }
}
}

// OpenMP MNK Offload
extern "C" {
void matmult_mnk_offload(int m, int n, int k, double **A, double **B, double **C) {
    double h2d_time, d2h_time, kernel_time;  // Timing variables

    initialize_offload(m, n, k, A, B, C, &h2d_time);  // Start GPU offload

    double start_kernel = omp_get_wtime();

    // Perform matrix multiplication on GPU
    #pragma omp target teams distribute parallel for collapse(2) num_teams(min(256, m)) thread_limit(128)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int l = 0; l < k; l++) {
                sum += A[i][l] * B[l][j];
            }
            C[i][j] = sum;
        }
    }

    kernel_time = omp_get_wtime() - start_kernel;

    finalize_offload(m, n, k, A, B, C, &d2h_time);  // Retrieve results from GPU

    // Ensure only master thread prints results
    #pragma omp master
    {
        #pragma omp flush
        printf("mnk_offload, %d, %f, %f, %f\n", m, h2d_time, kernel_time, d2h_time);
    }
}
}

#define _BLOCK_SIZE 64  // Fixed block size for optimization
#define _ELEMENTS_PER_THREAD 2

extern "C" {
void matmult_blk_offload(int m, int n, int k, double **A, double **B, double **C) {
    
    double h2d_time, d2h_time, kernel_time;

    // Perform H2D transfer
    initialize_offload(m, n, k, A, B, C, &h2d_time);
    
    // Start Kernel Execution Timing
    double start_kernel = omp_get_wtime();

        #pragma omp target teams distribute parallel for collapse(2) num_teams(m / _BLOCK_SIZE) thread_limit(128)
    for (int i1 = 0; i1 < m; i1 += _BLOCK_SIZE) {
        for (int j = 0; j < n; j++) {
            double temp_sum[_BLOCK_SIZE] = {0};

            for (int l = 0; l < k; l++) {
                // Compute multiple elements per thread
                #pragma omp simd  // Vectorization for efficiency
                for (int i2 = 0; i2 < _BLOCK_SIZE; i2 += _ELEMENTS_PER_THREAD) {
                    for (int e = 0; e < _ELEMENTS_PER_THREAD; e++) {
                        if (i1 + i2 + e < m) {  // Prevent out-of-bounds access
                            temp_sum[i2 + e] += A[i1 + i2 + e][l] * B[l][j];
                        }
                    }
                }
            }

            for (int i2 = 0; i2 < _BLOCK_SIZE; i2++) {
                if (i1 + i2 < m) {
                    C[i1 + i2][j] = temp_sum[i2];
                }
            }
        }
    }


    #pragma omp taskwait
    double end_kernel = omp_get_wtime();
    kernel_time = end_kernel - start_kernel;

    // Perform D2H transfer
    finalize_offload(m, n, k, A, B, C, &d2h_time);

    #pragma omp master
    {
        #pragma omp flush
        printf("blk_offload, %d, %f, %f, %f\n", m, h2d_time, kernel_time, d2h_time);
    }

}
}


#define SLABS 2  // Number of slabs (must evenly divide m)

// OpenMP Asynchronous Offload
extern "C" {
void matmult_asy_offload(int m, int n, int k, double **A, double **B, double **C) {
    if (m % SLABS != 0) {
        printf("Error: m (%d) must be divisible by SLABS (%d)!\n", m, SLABS);
        return;
    }

    int slab_size = m / SLABS;  // Compute slab size
    double h2d_time, d2h_time, kernel_time;

    // Perform H2D transfer
    initialize_offload(m, n, k, A, B, C, &h2d_time);

    // Start Kernel Execution Timing
    double start_kernel = omp_get_wtime();

    #pragma omp parallel for
    for (int s = 0; s < SLABS; ++s) {
        int start = s * slab_size;
        int length = slab_size;

        #pragma omp target update to(A[start:length][0:k]) nowait depend(out:A)

        #pragma omp target teams distribute parallel for collapse(2) \
            num_teams(min(m,256)) thread_limit(128) \
            map(to:A[start:length][0:k]) \
            map(tofrom:C[start:length][0:n]) \
            depend(in:A) depend(out:C) nowait
        for (int i = start; i < start + length; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int l = 0; l < k; l++) {
                    sum += A[i][l] * B[l][j];
                }
                C[i][j] = sum;
            }
        }

        #pragma omp target update from(C[start:length][0:n]) depend(in:C) nowait
    }

    #pragma omp taskwait
    double end_kernel = omp_get_wtime();
    kernel_time = end_kernel - start_kernel;

    // Perform D2H transfer
    finalize_offload(m, n, k, A, B, C, &d2h_time);

    #pragma omp master
    {
        #pragma omp flush
        printf("asy_offload, %d, %f, %f, %f\n", m, h2d_time, kernel_time, d2h_time);
    }
}
}