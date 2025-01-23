#include <stdio.h>
#include <omp.h>

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

#define NUM_SLABS 8  // Number of slabs
#define BLOCK_SIZE 8  // Block size for efficient memory access

extern "C" {
void matmult_asy_offload(int m, int n, int k, double* A, double* B, double* C) {
    int slab_size = m / NUM_SLABS;
    double h2d_time = 0.0, kernel_time = 0.0, d2h_time = 0.0;

    // **Fully allocate matrices on GPU**
    #pragma omp target enter data map(alloc: A[0:m*k], B[0:k*n], C[0:m*n])
    
    // **Ensure `B` is fully copied to the device once**
    double start_H2D = omp_get_wtime();
    #pragma omp target update to(B[0:k*n])
    h2d_time += omp_get_wtime() - start_H2D;

    for (int slab = 0; slab < NUM_SLABS; slab++) {
        int start = slab * slab_size;
        int end = (slab + 1) * slab_size;

        // **Async H2D transfer for each slab**
        start_H2D = omp_get_wtime();
        #pragma omp target update to(A[start * k:(end - start) * k]) nowait
        h2d_time += omp_get_wtime() - start_H2D;

        double start_kernel = omp_get_wtime();
        #pragma omp target teams distribute parallel for collapse(2) nowait \
                map(to: A[start * k:(end - start) * k]) map(tofrom: C[start * n:(end - start) * n]) \
                num_teams(128) thread_limit(64)
        for (int i1 = start; i1 < end; i1 += BLOCK_SIZE) {
            for (int j = 0; j < n; j++) {
                if (BLOCK_SIZE <= (end - i1)) {
                    double temp_sum[BLOCK_SIZE] = {0};
                    for (int l = 0; l < k; l++) {   
                        for (int i2 = 0; i2 < BLOCK_SIZE; i2++) {
                            temp_sum[i2] += A[(i1 + i2) * k + l] * B[l * n + j];
                        }
                    }
                    for (int i2 = 0; i2 < BLOCK_SIZE; i2++) {
                        C[(i1 + i2) * n + j] = temp_sum[i2];
                    }
                } else { 
                    for (int i2 = 0; i2 < (end - i1); i2++) {
                        double sum = 0.0;
                        for (int l = 0; l < k; l++) {   
                            sum += A[(i1 + i2) * k + l] * B[l * n + j];
                        }
                        C[(i1 + i2) * n + j] = sum;
                    }
                }
            }
        }
        kernel_time += omp_get_wtime() - start_kernel;

        // **Async D2H transfer for each slab**
        double start_D2H = omp_get_wtime();
        #pragma omp target update from(C[start * n:(end - start) * n]) nowait
        d2h_time += omp_get_wtime() - start_D2H;
    }

    #pragma omp taskwait

    // **Ensure COMPLETE DATA REMOVAL AFTER COMPUTATION**
    #pragma omp target exit data map(delete: A[0:m*k], B[0:k*n], C[0:m*n])

    // **Final print of transfer and computation times**
    printf("asy_offload, %d, %f, %f, %f\n", m, h2d_time, kernel_time, d2h_time);
}
}
