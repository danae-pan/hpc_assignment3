/* jacobi.c - Poisson problem in 3d
 *
 */
#include "jacobi_offload.h"
#include <stdlib.h>
#include <stdio.h>
#include "d_alloc3d.h"
#include "alloc3d.h"
#include <math.h>
#include <omp.h>

int jacobi_offload(double ***f, double ***u, double ***u_new, int N, int iter_max)
{
    double h2 = (2.0 / N) * (2.0 / N);
    double *f_data = f[0][0];
    double *u_data = u[0][0];
    double *u_new_data = u_new[0][0];

    int teams = (N * N) / 128;  // Scale teams dynamically based on grid size
    teams = (teams < 32) ? 32 : teams;  // Ensure a minimum of 32 teams

    #pragma omp target
    {
        printf("DEBUG: Checking OpenMP Offload: Running on Device ID %d\n", omp_is_initial_device());
    }

    for (int iter = 0; iter < iter_max; iter++){
        #pragma omp target teams num_teams(teams) thread_limit(128) distribute parallel for
        for (int i = 1; i < N + 1; i++) {
            for (int j = 1; j < N + 1; j++) {
                for (int k = 1; k < N + 1; k++) {
                    int index = i * (N+2) * (N+2) + j * (N+2) + k;
                    u_new_data[index] = (1.0 / 6.0) * (u_data[index-1] + u_data[index+1] +
                                                       u_data[index - (N+2)] + u_data[index + (N+2)] +
                                                       u_data[index - (N+2)*(N+2)] + u_data[index + (N+2)*(N+2)] +
                                                       h2 * f_data[index]);
                }
            }
        }

        // Swap pointers
        double *temp = u_data;
        u_data = u_new_data;
        u_new_data = temp;
    }
    
    return iter_max;
}

int jacobi_offload_dual(double *f, double *u, double *u_new, int N, int iter_max, int device_id)
{
    double h2 = (2.0 / N) * (2.0 / N);

    int teams = (N * N) / 128;  // Scale teams dynamically based on grid size
    teams = (teams < 32) ? 32 : teams;  // Ensure a minimum of 32 teams

    omp_set_default_device(device_id);

    #pragma omp target device(device_id)
    {
        printf("DEBUG: Running on Device ID %d\n", device_id);
    }

    for (int iter = 0; iter < iter_max; iter++) {
        #pragma omp target teams num_teams(teams) thread_limit(128) distribute parallel for
        for (int i = 1; i < N + 1; i++) {
            for (int j = 1; j < N + 1; j++) {
                for (int k = 1; k < N + 1; k++) {
                    int index = i * (N+2) * (N+2) + j * (N+2) + k;
                    u_new[index] = (1.0 / 6.0) * (u[index-1] + u[index+1] +
                                                  u[index - (N+2)] + u[index + (N+2)] +
                                                  u[index - (N+2)*(N+2)] + u[index + (N+2)*(N+2)] +
                                                  h2 * f[index]);
                }
            }
        }

        // Synchronize boundary data between GPUs
        size_t boundary_size = (N+2) * (N+2) * sizeof(double);
        double *host_buffer = (double *)malloc(boundary_size);  // Allocate buffer on host

        if (device_id == 0) {
            // Copy boundary from GPU 0 to host
            omp_target_memcpy(host_buffer, u_new + ((N/2 - 1) * (N+2) * (N+2)),
                            boundary_size, 0, 0, omp_get_initial_device(), 0);

            // Copy boundary from host to GPU 1
            omp_target_memcpy(u_new + ((N/2) * (N+2) * (N+2)), host_buffer,
                            boundary_size, 0, 0, 1, omp_get_initial_device());
        } else {
            // Copy boundary from GPU 1 to host
            omp_target_memcpy(host_buffer, u_new + ((N/2) * (N+2) * (N+2)),
                            boundary_size, 0, 0, omp_get_initial_device(), 1);

            // Copy boundary from host to GPU 0
            omp_target_memcpy(u_new + ((N/2 - 1) * (N+2) * (N+2)), host_buffer,
                            boundary_size, 0, 0, 0, omp_get_initial_device());
        }

        free(host_buffer);  // Free host buffer

        // Swap pointers
        double *temp = u;
        u = u_new;
        u_new = temp;
    }
    
    return iter_max;
}


int jacobi_offload_norm(double ***f, double ***u, double ***u_new, int N, int iter_max, double tolerance) 
{
    double h2 = (2.0 / N) * (2.0 / N);
    double diff = 0.0;
    double *f_data = f[0][0];
    double *u_data = u[0][0];
    double *u_new_data = u_new[0][0];

    int teams = (N * N) / 128;  
    teams = (teams < 32) ? 32 : teams;  

    #pragma omp target
    {
        printf("DEBUG: Checking OpenMP Offload: Running on Device ID %d\n", omp_is_initial_device());
    }

    for (int iter = 0; iter < iter_max; iter++) {
        diff = 0.0;

        #pragma omp target teams num_teams(teams) thread_limit(128) distribute parallel for reduction(+:diff)
        for (int i = 1; i < N + 1; i++) {
            for (int j = 1; j < N + 1; j++) {
                for (int k = 1; k < N + 1; k++) {
                    int index = i * (N+2) * (N+2) + j * (N+2) + k;
                    
                    double new_val = (1.0 / 6.0) * (u_data[index-1] + u_data[index+1] +
                                                    u_data[index - (N+2)] + u_data[index + (N+2)] +
                                                    u_data[index - (N+2)*(N+2)] + u_data[index + (N+2)*(N+2)] +
                                                    h2 * f_data[index]);

                    double diff_local = new_val - u_data[index];
                    diff += diff_local * diff_local; // Sum of squared differences
                    
                    u_new_data[index] = new_val;
                }
            }
        }

        // Compute the mean squared error and check stopping criterion
        diff = sqrt(diff / (N * N * N));

        printf("Iteration %d with mean squared difference = %.6f\n", iter + 1, diff);

        if (diff < tolerance) {
            printf("Converged after %d iterations with mean squared difference = %.6f\n", iter + 1, diff);
            break;
        }

        // Swap pointers
        double *temp = u_data;
        u_data = u_new_data;
        u_new_data = temp;
    }

    return iter_max;
}
