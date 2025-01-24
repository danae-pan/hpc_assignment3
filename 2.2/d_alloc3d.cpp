#include <stdlib.h>
#include <omp.h>
#include "d_alloc3d.h"
#include <stdio.h>

double ***d_malloc_3d(int m, int n, int k, double** data){
    if (m <= 0 || n <= 0 || k <= 0)
        return NULL;

    // Allocate host pointer structure
    double ***p = (double***) malloc(m * sizeof(double **) + m * n * sizeof(double *));
    if (p == NULL) {
        printf("ERROR: Host memory allocation for p failed!\n");
        return NULL;
    }

    for (int i = 0; i < m; i++) {
        p[i] = (double **) (p + m) + i * n;
    }


    // Allocate actual data on the GPU
    double *a = (double*) omp_target_alloc(m * n * k * sizeof(double), omp_get_default_device());
    if (a == NULL) {
        printf("ERROR: GPU memory allocation for a failed!\n");
        free(p);
        return NULL;
    }

    // Assign memory structure on the host (CPU)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            p[i][j] = a + (i * n * k) + (j * k);
        }
    }

    *data = a; // Return device pointer
    return p;  // Return host pointer 
}

void d_free_3d(double* data) {
    omp_target_free(data, omp_get_default_device());
}