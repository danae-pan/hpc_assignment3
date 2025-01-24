#ifndef __D_ALLOC_3D
#define __D_ALLOC_3D


#ifdef __cplusplus
extern "C"{
#endif

double*** d_malloc_3d(int m, int n, int k, double** data);
void d_free_3d(double* data);


#ifdef __cplusplus
}
#endif

#endif /* __D_ALLOC_3D */
