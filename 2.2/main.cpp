#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>  
#include "d_alloc3d.h"
#include "alloc3d.h"
#include "jacobi_offload.h"
#include "jacobi.h"
#include "warmup.h"

int main(int argc, char *argv[]) {
    int N = 100, iter_max = 1000, method = 0; // Default values
    double start_T = 20.0; // Default starting temperature
    double ***f, ***u, ***u_new;
    double *f_gpu0, *u_gpu0, *u_new_gpu0;
    double *f_gpu1, *u_gpu1, *u_new_gpu1;

    // Check for command-line arguments
    if (argc >= 4) {
        N = atoi(argv[1]);        // Grid size
        iter_max = atoi(argv[2]); // Maximum iterations
        method = atoi(argv[3]);  // 0 = one GPU, 1 = dual GPU
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

    warm_up();  // Warm up the GPU before computation

    // Run the chosen Jacobi method
    //printf("DEBUG: Running Jacobi Offload computation...\n");
    double start_time = omp_get_wtime();

    if (method == 0) {
        // Single GPU execution
        size_t size = (N+2) * (N+2) * (N+2) * sizeof(double);
        d_malloc_3d(N+2, N+2, N+2, &f_gpu0);
        d_malloc_3d(N+2, N+2, N+2, &u_gpu0);
        d_malloc_3d(N+2, N+2, N+2, &u_new_gpu0);

        omp_target_memcpy(f_gpu0, f[0][0], size, 0, 0, 0, omp_get_initial_device());
        omp_target_memcpy(u_gpu0, u[0][0], size, 0, 0, 0, omp_get_initial_device());
        omp_target_memcpy(u_new_gpu0, u_new[0][0], size, 0, 0, 0, omp_get_initial_device());

        jacobi_offload(f, u, u_new, N, iter_max);

        omp_target_memcpy(u_new[0][0], u_new_gpu0, size, 0, 0, omp_get_initial_device(), 0);

        d_free_3d(f_gpu0);
        d_free_3d(u_gpu0);
        d_free_3d(u_new_gpu0);

    } else {
        // Dual GPU execution
        int N_half = N / 2;
        size_t size_half = (N_half+2) * (N+2) * (N+2) * sizeof(double);

        // Allocate GPU memory
        omp_set_default_device(0);
        d_malloc_3d(N_half+2, N+2, N+2, &f_gpu0);
        d_malloc_3d(N_half+2, N+2, N+2, &u_gpu0);
        d_malloc_3d(N_half+2, N+2, N+2, &u_new_gpu0);

        omp_set_default_device(1);
        d_malloc_3d(N_half+2, N+2, N+2, &f_gpu1);
        d_malloc_3d(N_half+2, N+2, N+2, &u_gpu1);
        d_malloc_3d(N_half+2, N+2, N+2, &u_new_gpu1);

        // Enable Peer Access
        cudaSetDevice(0);
        cudaDeviceEnablePeerAccess(1, 0);
        cudaSetDevice(1);
        cudaDeviceEnablePeerAccess(0, 0);

        // Copy data to GPUs
        omp_set_default_device(0);
        omp_target_memcpy(f_gpu0, f[0][0], size_half, 0, 0, 0, omp_get_initial_device());
        omp_target_memcpy(u_gpu0, u[0][0], size_half, 0, 0, 0, omp_get_initial_device());

        omp_set_default_device(1);
        omp_target_memcpy(f_gpu1, f[N_half][0], size_half, 0, 0, 1, omp_get_initial_device());
        omp_target_memcpy(u_gpu1, u[N_half][0], size_half, 0, 0, 1, omp_get_initial_device());

        // Run Jacobi on both GPUs in parallel
        #pragma omp parallel sections
        {
            #pragma omp section
            jacobi_offload_dual(f_gpu0, u_gpu0, u_new_gpu0, N_half, iter_max, 0);

            #pragma omp section
            jacobi_offload_dual(f_gpu1, u_gpu1, u_new_gpu1, N_half, iter_max, 1);
        }

        
        // Copy results back to CPU
        omp_target_memcpy(u_new[0][0], u_new_gpu0, size_half, 0, 0, omp_get_initial_device(), 0);
        omp_target_memcpy(u_new[N_half][0], u_new_gpu1, size_half, 0, 0, omp_get_initial_device(), 1);
    }
    double end_time = omp_get_wtime();
    printf("Jacobi Offload Execution Time: %.6f seconds\n", end_time - start_time);

    return 0;
}