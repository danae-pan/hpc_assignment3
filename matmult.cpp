#include <stdio.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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

// OpenMP MKN
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


// lib
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

// OpenMP MKN Offload
extern "C" {
void matmult_mkn_offload(int m, int n, int k, double **A, double **B, double **C) {
    double h2d_time, d2h_time, kernel_time;
    int num_teams = omp_get_num_teams();

    initialize_offload(m, n, k, A, B, C, &h2d_time);

    double start_kernel = omp_get_wtime();

    #pragma omp target teams distribute parallel for collapse(2) thread_limit(32)
    for (int i = 0; i < m; i++) {
        for (int l = 0; l < k; l++) {
            for (int j = 0; j < n; j++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }

    kernel_time = omp_get_wtime() - start_kernel;

    finalize_offload(m, n, k, A, B, C, &d2h_time);

    #pragma omp master
    {
        #pragma omp flush
        printf("mkn_offload, %d, %d, %f, %f, %f\n", m, num_teams, h2d_time, kernel_time, d2h_time);
    }
}
}

// OpenMP MNK Offload
extern "C" {
void matmult_mnk_offload(int m, int n, int k, double **A, double **B, double **C) {
    double h2d_time, d2h_time, kernel_time;  // Timing variables
    int num_teams = omp_get_num_teams();

    initialize_offload(m, n, k, A, B, C, &h2d_time);  // Start GPU offload

    double start_kernel = omp_get_wtime();

    // Perform matrix multiplication on GPU
    #pragma omp target teams distribute parallel for collapse(2) thread_limit(32)
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
        printf("mnk_offload, %d, %d, %f, %f, %f\n", m, num_teams, h2d_time, kernel_time, d2h_time);
    }
}
}


#define SLAPS 4  // Number of slabs (must evenly divide m)
#define BLOCK_SIZE 64  // Block size for parallel execution

extern "C" {
void matmult_asy_offload(int m, int n, int k, double **A, double **B, double **C) {
    // Ensure matrix sizes are divisible by SLAPS
    if (m % SLAPS != 0) {
        printf("Error: m (%d) must be divisible by SLAPS (%d)!\n", m, SLAPS);
        return;
    }

    int slab_size = m / SLAPS;  // Compute slab size

    // Start timing H2D transfer
    double start_H2D = omp_get_wtime();

    // Allocate and transfer B once (persists on device)
    #pragma omp target data map(alloc:A[0:m][0:k], B[0:k][0:n], C[0:m][0:n])
    {
        #pragma omp target update to(B[0:k][0:n]) // Move B to GPU

        // End timing H2D transfer
        double end_H2D = omp_get_wtime();
        double h2d_time = end_H2D - start_H2D;

        // Start parallel execution of slabs
        #pragma omp parallel for
        for (int s = 0; s < SLAPS; ++s) {
            int start = s * slab_size;  // Starting index for the slab
            int length = slab_size;     // Number of rows in the slab

            // Transfer slab A to GPU asynchronously
            #pragma omp target update to(A[start:length][0:k]) nowait depend(out:A)

            // Perform matrix multiplication asynchronously
            #pragma omp target teams distribute parallel for collapse(2) \
                num_teams(128) thread_limit(64) \
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

            // Transfer results back to CPU asynchronously
            #pragma omp target update from(C[start:length][0:n]) depend(in:C) nowait
        }

        // Ensure all slabs complete before proceeding
        #pragma omp taskwait

        // Start timing D2H transfer
        double start_D2H = omp_get_wtime();
        #pragma omp target exit data map(delete:A[0:m][0:k], B[0:k][0:n], C[0:m][0:n])
        double end_D2H = omp_get_wtime();
        double d2h_time = end_D2H - start_D2H;

        // Print timing results
        printf("asy_offload, %d, %f, %f, %f\n", m, h2d_time, omp_get_wtime() - start_H2D, d2h_time);
    }
}
}


// OpenMP LIB Offload using CUBLAS
extern "C" {
void matmult_lib_offload(int m, int n, int k, double **A, double **B, double **C) {
    double h2d_time, d2h_time, kernel_time;

    double *d_A, *d_B, *d_C;
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate memory on GPU
    double start_H2D = omp_get_wtime();
    cudaMalloc((void **)&d_A, m * k * sizeof(double));
    cudaMalloc((void **)&d_B, k * n * sizeof(double));
    cudaMalloc((void **)&d_C, m * n * sizeof(double));

    // Transfer A and B to GPU
    cudaMemcpy(d_A, *A, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, *B, k * n * sizeof(double), cudaMemcpyHostToDevice);
    h2d_time = omp_get_wtime() - start_H2D;

    double alpha = 1.0, beta = 0.0;

    // Start kernel execution time measurement
    double start_kernel = omp_get_wtime();
    
    // Perform DGEMM on GPU
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                 m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);
    
    kernel_time = omp_get_wtime() - start_kernel;

    // Transfer C back to host
    double start_D2H = omp_get_wtime();
    cudaMemcpy(*C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);
    d2h_time = omp_get_wtime() - start_D2H;

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    // Print results in the same format
    #pragma omp master
    {
        #pragma omp flush
        printf("lib_offload, %d, %f, %f, %f\n", m, h2d_time, kernel_time, d2h_time);
    }
}
}