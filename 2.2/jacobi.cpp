/* jacobi.c - Poisson problem in 3d
 *
 */
#include "jacobi.h"
#include <stdlib.h>
#include <stdio.h>
#include "alloc3d.h"
#include <math.h>
#include <omp.h>


int jacobi_parallel_opt(double ***f, double ***u, double ***u_new, int N, int iter_max)
{
    double h = 2.0 / N;
    double h2 = h * h;
    int iter;

    printf("DEBUG: Checking OpenMP Execution: Running on Device ID %d\n", omp_is_initial_device());

    for  (iter = 1; iter <= iter_max; iter++)
    {
        #pragma omp parallel for shared(f, u, u_new, h2) schedule(static)
        for (int i = 1; i <= N; i++)
        {
            for (int j = 1; j <= N; j++)
            {
                for (int k = 1; k <= N; k++)
                {
                    double temp = (1.0 / 6.0) * (u[i - 1][j][k] + u[i + 1][j][k] +
                                                 u[i][j - 1][k] + u[i][j + 1][k] +
                                                 u[i][j][k - 1] + u[i][j][k + 1] +
                                                 h2 * f[i][j][k]);

                    // Update grid point
                    u_new[i][j][k] = temp;
                }
            }
        }

        // Swap u and u_new
        double ***temp = u;
        u = u_new;
        u_new = temp;
    }
    
    return iter_max; 
}


int jacobi_parallel_opt_norm(double ***f, double ***u, double ***u_new, int N, int iter_max, double *tolerance, double* final_diff)
{
    double h = 2.0 / N;
    double h2 = h * h;
    double diff = 0.0;
    int iter;

    for  (iter = 1; iter <= iter_max; iter++)
    {
        diff = 0.0;
        #pragma omp parallel for shared(f, u, u_new, h2) reduction(+:diff) schedule(static)
        for (int i = 1; i <= N; i++)
        {
            for (int j = 1; j <= N; j++)
            {
                for (int k = 1; k <= N; k++)
                {
                    double temp = (1.0 / 6.0) * (u[i - 1][j][k] + u[i + 1][j][k] +
                                                 u[i][j - 1][k] + u[i][j + 1][k] +
                                                 u[i][j][k - 1] + u[i][j][k + 1] +
                                                 h2 * f[i][j][k]);

                    double local_diff = temp - u[i][j][k];  // Compute difference

                    // Update grid point
                    u_new[i][j][k] = temp;

                    // Accumulate squared differences for RMS computation
                    diff += local_diff * local_diff;
                }
            }
        }

        // Compute RMS norm
        diff = sqrt(diff / (N * N * N));

        // Stop if converged
        if (diff < *tolerance)
        {
            printf("Converged after %d iterations with RMS diff = %.6f\n", iter, diff);
            *final_diff = diff;  // Store final diff value
            return iter;
        }

        // Swap u and u_new
        double ***temp = u;
        u = u_new;
        u_new = temp;
    }

    *final_diff = diff;  // Store final diff value even if max_iter is reached
    return iter_max; // Return max_iter if no convergence
}
