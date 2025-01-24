/* jacobi.h - Poisson problem 
 *
 * $Id: jacobi.h,v 1.1 2006/09/28 10:12:58 bd Exp bd $
 */

#ifndef _JACOBI_H
#define _JACOBI_H

#ifdef __cplusplus
extern "C"{
#endif

int jacobi_parallel_opt(double ***, double ***, double ***, int, int);

int jacobi_parallel_opt_norm(double ***, double ***, double ***, int, int, double, double*);

#ifdef __cplusplus
}
#endif

#endif
