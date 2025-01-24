/* jacobi.h - Poisson problem 
 *
 * $Id: jacobi.h,v 1.1 2006/09/28 10:12:58 bd Exp bd $
 */

#ifndef _JACOBI_H_GPU
#define _JACOBI_H_GPU

#ifdef __cplusplus
extern "C"{
#endif

int jacobi_offload(double ***, double ***, double ***, int, int);
int jacobi_offload_dual(double *, double *, double *, int, int, int);
int jacobi_offload_norm(double ***, double ***, double ***, int, int, double); 

#ifdef __cplusplus
}
#endif


#endif
