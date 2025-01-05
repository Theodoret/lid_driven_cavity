#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>

void write(std::string name, float *matrix, int Nx, int Ny);

__global__
void boundary(float dy, float dx, int Nx, int Ny,
            float lid,
            float *psi, float *omega);

__global__
void simple_boundary(float dy, float dx, int Nx, int Ny,
            float lid,
            float *omega, float *u, float *v);

__global__
void vorticity(float dy, float dx, float dt, int Nx, int Ny,
            float Re, 
            float *psi, float *omega, float *u, float *v,
            float *psi_new);

__global__
void stream(float dy, float dx, int Nx, int Ny,
            float Nmax_SOR_psi, float psi_Toll, 
            float *psi, float *omega,
            float *psi_new, float *psi_SOR,
            float *psi_SORerr);

__global__
void velocity(float dy, float dx, int Nx, int Ny,
            float *psi, float *u, float *v,
            float *psi_new);