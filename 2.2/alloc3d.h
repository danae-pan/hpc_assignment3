#ifndef __ALLOC_3D
#define __ALLOC_3D

#ifdef __cplusplus
extern "C"{
#endif

double ***malloc_3d(int m, int n, int k);
void free_3d(double ***array3D);


#ifdef __cplusplus
}
#endif

#endif /* __ALLOC_3D */
