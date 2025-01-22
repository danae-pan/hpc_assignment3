#include <stdio.h>

extern "C" {
    #include <omp.h>
    
    

    # pragma omp parallel for 
    void matmult_mkn(int m,int n,int k,double **A,double **B,double **C) {
        double start = omp_get_wtime();
        for(int i = 0; i < m*n; i++) (*C)[i] = 0;

        for(int row = 0; row < m; row++) {
            for(int i = 0; i < k; i++) {
                for(int col = 0; col < n; col++) {
                    C[row][col] += A[row][i] * B[i][col];
                }
            }
        }
        double end = omp_get_wtime();
        printf("Execution time %f", end - start);
    }

    
    
}