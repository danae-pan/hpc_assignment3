#include <stdio.h>

extern "C" {
#include <cblas.h>
#include <omp.h>

    void matmult_lib(int m,int n,int k,double **A,double **B,double **C) {
        double start = omp_get_wtime();
        /*
        C = alpha * A * B + beta * C
        101: Row-major order
        111: No transpose for A
        111: No transpose for B
        The leading dimension for A is k
        The leading dimension for B is m
        The leading dimension for C is m
        */
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, *A, k, *B, n, 0.0, *C, n);
        double end = omp_get_wtime();
        printf("Execution time %f\n", end - start);
    }
}