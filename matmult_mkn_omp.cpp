#include <stdio.h>

extern "C" {
    #include <omp.h>
    
    void matmult_mkn_omp(int m,int n,int k,double **A,double **B,double **C) {
        double start = omp_get_wtime();
        #pragma omp parallel for 
        for(int row = 0; row < m; row++) {
            for(int col = 0; col < n; col++) {
                C[row][col] = 0;
                for(int i = 0; i < k; i++) {  
                    C[row][col] += A[row][i] * B[i][col];
                }
            }
        }
        double end = omp_get_wtime();
        printf("Execution time %f", end - start);
    }


    
    
}