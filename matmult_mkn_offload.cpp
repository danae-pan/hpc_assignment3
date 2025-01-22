#include <stdio.h>
#include <omp.h>

// OpenMP MKN Offload
void matmult_mkn_offload(int m, int n, int k, double **A, double **B, double **C) {
    double start = omp_get_wtime();

    // Initialize C to zero
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;
        }
    }

    // Transfer data to device
    #pragma omp target data map(to: A[0:m][0:k], B[0:k][0:n]) map(tofrom: C[0:m][0:n])
    {
        #pragma omp target teams distribute parallel for collapse(2) num_teams(_TEAMS) thread_limit(_THREADS)
        for (int i = 0; i < m; i++) {
            for (int l = 0; l < k; l++) {
                for (int j = 0; j < n; j++) {
                    C[i][j] += A[i][l] * B[l][j];  // Potential race condition
                }
            }
        }
    }
    
    double end = omp_get_wtime();
    printf("MKN Offload Execution Time: %f seconds\n", end - start);
}

// OpenMP MNK Offload
void matmult_mnk_offload(int m, int n, int k, double **A, double **B, double **C) {
    double start = omp_get_wtime();

    // Initialize C to zero
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;
        }
    }

    // Transfer data to device
    #pragma omp target data map(to: A[0:m][0:k], B[0:k][0:n]) map(tofrom: C[0:m][0:n])
    {
        #pragma omp target teams distribute parallel for collapse(2) num_teams(_TEAMS) thread_limit(_THREADS)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int l = 0; l < k; l++) {
                    sum += A[i][l] * B[l][j];
                }
                C[i][j] = sum;
            }
        }
    }

    double end = omp_get_wtime();
    printf("MNK Offload Execution Time: %f seconds\n", end - start);
}
