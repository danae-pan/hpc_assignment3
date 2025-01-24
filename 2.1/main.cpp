#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "d_alloc3d.h"
#include "alloc3d.h"
#include "jacobi_offload.h"
#include "jacobi.h"
#include "warmup.h"

int main(int argc, char *argv[]) {
    int N = 100, iter_max = 1000, method = 0; // Default values
    double start_T = 20.0; // Default starting temperature
    double ***f, ***u, ***u_new;
    double *f_gpu, *u_gpu, *u_new_gpu;

    // Check for command-line arguments
    if (argc >= 4) {
        N = atoi(argv[1]);        // Grid size
        iter_max = atoi(argv[2]); // Maximum iterations
        method = atoi(argv[3]);  // 0 = map, 1 = memcpy, 2 = CPU
    }

    // Allocate memory on the CPU using malloc_3d()
    //printf("DEBUG: Allocating memory on CPU...\n");
    f = malloc_3d(N+2, N+2, N+2);
    u = malloc_3d(N+2, N+2, N+2);
    u_new = malloc_3d(N+2, N+2, N+2);

    // Check if memory allocation was successful
    if (!f || !u || !u_new) {
        printf("ERROR: Memory allocation failed!\n");
        return 1;
    }

    //printf("DEBUG: Initializing grid data on CPU...\n");

    double delta = 2.0 / (N + 1);

    for (int i = 0; i < N + 2; i++) {
        for (int j = 0; j < N + 2; j++) {
            for (int k = 0; k < N + 2; k++) {
                // Boundary condition
                if (i == 0 || i == N + 1 || j == 0 || j == N + 1 || k == 0 || k == N + 1) {
                    u[i][j][k] = start_T;
                } else {
                    u[i][j][k] = 0.0;
                }
                u_new[i][j][k] = u[i][j][k];

                // Define heat source (Radiator)
                if (i >= N * 0.125 && i <= N * 0.375 &&
                    j >= N * 0.25 && j <= N * 0.5 &&
                    k >= N * 0.333 && k <= N * 0.666) {
                    f[i][j][k] = 200.0;
                } else {
                    f[i][j][k] = 0.0;
                }
            }
        }
    }
    //printf("DEBUG: Grid initialization on CPU complete.\n");


    // Allocate GPU Memory using d_malloc_3d()
    //printf("DEBUG: Allocating memory on GPU...\n");
    d_malloc_3d(N+2, N+2, N+2, &f_gpu);
    d_malloc_3d(N+2, N+2, N+2, &u_gpu);
    d_malloc_3d(N+2, N+2, N+2, &u_new_gpu);

    //printf("DEBUG: Running CUDA warm-up...\n");
    warm_up();  // Warm up the GPU before computation
    //printf("DEBUG: CUDA warm-up complete.\n");

    // Run the chosen Jacobi method
    //printf("DEBUG: Running Jacobi Offload computation...\n");
    double start_time = omp_get_wtime();

    if (method == 0) {
        // Use OpenMP map-based offloading
        //printf("DEBUG: Copying grid data to GPU using OpenMP map...\n");
        #pragma omp target enter data map(to: f[0:N+2][0:N+2][0:N+2], \
                                         u[0:N+2][0:N+2][0:N+2], \
                                         u_new[0:N+2][0:N+2][0:N+2])
        //printf("DEBUG: Grid data copied to GPU successfully using map.\n");

        jacobi_offload(f, u, u_new, N, iter_max);

        // Copy results back to CPU (only needed for map)
        #pragma omp target exit data map(from: u_new[0:N+2][0:N+2][0:N+2])

    } else if(method == 1){
        // Use explicit "omp_target_memcpy()"
        //printf("DEBUG: Copying grid data to GPU using omp_target_memcpy...\n");
        size_t size = (N+2) * (N+2) * (N+2) * sizeof(double);

        omp_target_memcpy(f_gpu, f[0][0], size, 0, 0, omp_get_default_device(), omp_get_initial_device());
        omp_target_memcpy(u_gpu, u[0][0], size, 0, 0, omp_get_default_device(), omp_get_initial_device());
        omp_target_memcpy(u_new_gpu, u_new[0][0], size, 0, 0, omp_get_default_device(), omp_get_initial_device());

        //printf("DEBUG: Grid data copied to GPU successfully using omp_target_memcpy.\n");

        jacobi_offload(f, u, u_new, N, iter_max);

        // Copy results back to CPU
        //printf("DEBUG: Copying results back from GPU using omp_target_memcpy...\n");
        omp_target_memcpy(u_new[0][0], u_new_gpu, size, 0, 0, omp_get_initial_device(), omp_get_default_device());
    } else {
        // Jacobi parallel form Assignment 2:
        jacobi_parallel_opt(f, u, u_new, N, iter_max);
    }

    double end_time = omp_get_wtime();
    printf("Jacobi Offload Execution Time: %.6f seconds\n", end_time - start_time);

    // Free memory
    //printf("DEBUG: Freeing allocated memory...\n");
    free_3d(f);
    free_3d(u);
    free_3d(u_new);
    d_free_3d(f_gpu);
    d_free_3d(u_gpu);
    d_free_3d(u_new_gpu);

    return 0;
}