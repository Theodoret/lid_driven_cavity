#include "header.cuh"
#include <math.h>
#include <fstream>

__global__
void boundary(float dy, float dx, int Nx, int Ny,
            float lid,
            float *psi, float *omega) {
    int x = blockIdx.x*blockDim.x + threadIdx.x,
        y = blockIdx.y*blockDim.y + threadIdx.y,
        i = x + y*blockDim.x*gridDim.x;

    // omega derivative with finite difference
    if (i>0 && i<Nx-1){ // bottom
        omega[i] = -2*psi[i + Nx]/pow(dy,2);
    } else if (i>(Ny-1)*Nx && i<Ny*Nx-1 ){ // top
        omega[i] = -2*psi[i - Nx]/pow(dy,2) - lid*2/dx;
    } else if (i%Nx == 0 && i>Nx-1 && i<(Ny-1)*Nx){ // left
        omega[i] = -2*psi[i + 1]/pow(dx,2);
    } else if (i%Nx == Nx-1 && i>Nx-1 && i<(Ny-1)*Nx){ // right
        omega[i] = -2*psi[i - 1]/pow(dx,2);
    }
    __syncthreads();
}

__global__
void simple_boundary(float dy, float dx, int Nx, int Ny,
            float lid,
            float *omega, float *u, float *v) {
    int x = blockIdx.x*blockDim.x + threadIdx.x,
        y = blockIdx.y*blockDim.y + threadIdx.y,
        i = x + y*blockDim.x*gridDim.x;

    // omega derivative with finite difference
    if (i>0 && i<Nx-1){ // bottom
        omega[i] = -(u[i+Nx] - u[i])/dy;
    } else if (i>(Ny-1)*Nx && i<Ny*Nx-1 ){ // top
        omega[i] = -(u[i] - u[i-Nx])/dy;
    } else if (i%Nx == 0 && i>Nx-1 && i<(Ny-1)*Nx){ // left
        omega[i] = (v[i+1] - v[i])/dx;
    } else if (i%Nx == Nx-1 && i>Nx-1 && i<(Ny-1)*Nx){ // right
        omega[i] = (v[i] - v[i-1])/dx;
    }
    __syncthreads();
}

__global__
void vorticity(float dy, float dx, float dt, int Nx, int Ny,
            float Re, 
            float *psi, float *omega, float *u, float *v,
            float *psi_new) {
    int x = blockIdx.x*blockDim.x + threadIdx.x,
        y = blockIdx.y*blockDim.y + threadIdx.y,
        i = x + y*blockDim.x*gridDim.x;

    // temporary parameters
    float omegax, omegay, omegaxx, omegayy, LHS, RHS;

    if (i > Nx && i < (Ny-1)*Nx && i%Nx != 0 && i%Nx != Nx-1){
        // omega derivative with finite difference
        omegax = (omega[i+1] - omega[i-1])/(2*dx);
        omegay = (omega[i+Nx] - omega[i-Nx])/(2*dy);
        omegaxx = (omega[i+1] - 2*omega[i] + omega[i-1])/pow(dx,2);
        omegayy = (omega[i+Nx] - 2*omega[i] + omega[i-Nx])/pow(dy,2);
        __syncthreads();

        // vorticity function
        LHS = u[i]*omegax + v[i]*omegay;
        RHS = (1/Re)*(omegaxx + omegayy);
        omega[i] = omega[i] + dt*(RHS - LHS);

        psi_new[i] = psi[i];
        __syncthreads();
    }
}

__global__
void stream(float dy, float dx, int Nx, int Ny,
            float Nmax_SOR_psi, float psi_Toll, 
            float *psi, float *omega,
            float *psi_new, float *psi_SOR,
            float *psi_SORerr) {
    int x = blockIdx.x*blockDim.x + threadIdx.x,
        y = blockIdx.y*blockDim.y + threadIdx.y,
        i = x + y*blockDim.x*gridDim.x;

    // temporary parameters
    // float psi_SORerr;

    if (i > Nx && i < (Ny-1)*Nx && i%Nx != 0 && i%Nx != Nx-1){
        // stream function
        for (int l = 0; l<Nmax_SOR_psi; l++){
            psi_SOR[i] = 0.25*(psi_new[i+1] + psi_new[i-1] + psi_new[i+Nx] + psi_new[i-Nx] + dx*dy*omega[i]);
            __syncthreads();
            
            // psi_SORerr = 0;
            psi_new[i] = psi_SOR[i];
            __syncthreads();
            *psi_SORerr += abs(psi_new[i] - psi[i]);
            __syncthreads();
            
            if (*psi_SORerr <= psi_Toll){
                return;
            }
        }
        __syncthreads();
    }
}

__global__
void velocity(float dy, float dx, int Nx, int Ny,
            float *psi, float *u, float *v,
            float *psi_new) {
    int x = blockIdx.x*blockDim.x + threadIdx.x,
        y = blockIdx.y*blockDim.y + threadIdx.y,
        i = x + y*blockDim.x*gridDim.x;

    if (i > Nx && i < (Ny-1)*Nx && i%Nx != 0 && i%Nx != Nx-1){
        // velocity
        psi[i] = psi_new[i];
        __syncthreads();
        u[i] = (psi[i+Nx] - psi[i-Nx])/(2*dy); // u = psiy
        v[i] = -(psi[i+1] - psi[i-1])/(2*dx); // u = -psix
        __syncthreads();
    }
}

void write(std::string name, float *matrix, int Nx, int Ny){
    std::string extension = ".txt";
    std::ofstream file_text(name + extension);
    for (int i = 0; i<Nx; i++){
        for (int j = 0; j<Ny; j++){
            if ( j != 0){
                file_text << ",";
            }
            file_text << matrix[j*Nx + i];
        }
        file_text << "\n";
    }
}