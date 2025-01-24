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

