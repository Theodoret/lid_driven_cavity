#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>

// __global__ 
// void GPU_Navier(float dy, float dx, float dt, int Nt, 
//                 float Nmax_SOR_psi, float psi_Toll, 
//                 float lid, float Re, 
//                 float *psi, float *omega, float *u, float *v,
//                 float *psi_new, float *psi_SOR){
//     int Nx = blockIdx.x*blockDim.x + threadIdx.x;
//     int Ny = blockIdx.y*blockDim.y + threadIdx.y;
//     int i = Nx + Ny*blockDim.x*gridDim.x;
    
//     // const int N = Nx*Ny;
//     // temporary parameters
//     float omegax, omegay, omegaxx, omegayy, LHS, RHS, psi_SORerr;

//     // Time looping
//     __syncthreads();
//     for (int k = 0; k<Nt; k++){
//         // imposing boundary on omega
//         if (i>0 && i<Nx-1){ // bottom
//             omega[i] = -2*psi[i + Nx]/pow(dy,2);
//         } else if (i>(Ny-1)*Nx && i<Ny*Nx-1 ){ // top
//             omega[i] = -2*psi[i - Nx]/pow(dy,2) - lid*2/dx;
//         } else if (i%Nx == 0 && i>Nx-1 && i<(Ny-1)*Nx){ // left
//             omega[i] = -2*psi[i + 1]/pow(dx,2);
//         } else if (i%Nx == Nx-1 && i>Nx-1 && i<(Ny-1)*Nx){ // right
//             omega[i] = -2*psi[i - 1]/pow(dx,2);
//         }
//         __syncthreads();

//         if (i > Nx && i < (Ny-1)*Nx && i%Nx != 0 && i%Nx != Nx-1){
//             // omega derivative with finite difference
//             omegax = (omega[i+1] - omega[i-1])/(2*dx);
//             omegay = (omega[i+Nx] - omega[i-Nx])/(2*dy);
//             omegaxx = (omega[i+1] - 2*omega[i] + omega[i-1])/pow(dx,2);
//             omegayy = (omega[i+Nx] - 2*omega[i] + omega[i-Nx])/pow(dy,2);
//             __syncthreads();

//             // vorticity function
//             LHS = u[i]*omegax + v[i]*omegay;
//             RHS = (1/Re)*(omegaxx + omegayy);
//             omega[i] = omega[i] + dt*(RHS - LHS);

//             psi_new[i] = psi[i];
//             __syncthreads();

//             // stream function
//             for (int l = 0; l<Nmax_SOR_psi; l++){
//                 psi_SOR[i] = 0.25*(psi_new[i+1] + psi_new[i-1] + psi_new[i+Nx] + psi_new[i-Nx] + dx*dy*omega[i]);
//                 __syncthreads();
                
//                 psi_SORerr = 0;
//                 psi_new[i] = psi_SOR[i];
//                 psi_SORerr += abs(psi_new[i] - psi[i]);
//                 __syncthreads();
                
//                 if (psi_SORerr <= psi_Toll){
//                     break;
//                 }
//             }
//             __syncthreads();

//             // velocity
//             psi[i] = psi_new[i];
//             __syncthreads();
//             u[i] = (psi[i+Nx] - psi[i-Nx])/(2*dy); // u = psiy
//             v[i] = -(psi[i+1] - psi[i-1])/(2*dx); // u = -psix
//             __syncthreads();
//         }
//     }
// }

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
            float *psi_new, float *psi_SOR) {
    int x = blockIdx.x*blockDim.x + threadIdx.x,
        y = blockIdx.y*blockDim.y + threadIdx.y,
        i = x + y*blockDim.x*gridDim.x;

    // temporary parameters
    float psi_SORerr;

    if (i > Nx && i < (Ny-1)*Nx && i%Nx != 0 && i%Nx != Nx-1){
        // stream function
        for (int l = 0; l<Nmax_SOR_psi; l++){
            psi_SOR[i] = 0.25*(psi_new[i+1] + psi_new[i-1] + psi_new[i+Nx] + psi_new[i-Nx] + dx*dy*omega[i]);
            __syncthreads();
            
            psi_SORerr = 0;
            psi_new[i] = psi_SOR[i];
            psi_SORerr += abs(psi_new[i] - psi[i]);
            __syncthreads();
            
            if (psi_SORerr <= psi_Toll){
                break;
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

int main(int argc, char *argv[]){
    std::cout << "-1";
    // Hyperparameters
    const int Nx = 51, Ny = 51, N = Nx*Ny, Nt = 10000;
    float lid = 1, Re = 1000,
        xmin = 0, xmax = 1, dx = (xmax - xmin)/(Nx - 1), 
        ymin = 0, ymax = 1, dy = (ymax - ymin)/(Ny - 1), 
        tmin = 0, tmax = 100, dt = (tmax - tmin)/Nt;
    float Nmax_SOR_psi = 200, psi_Toll = 0.01;

    std::cout << "0";
    // domain matrix
    std::vector<float> x, y;
    for (int i = 0; i<Nx; i++){
        x.push_back(xmin + dx*i);
    }
    for (int i = 0; i<Ny; i++){
        y.push_back(ymin + dy*i);
    }
    float *X, *Y;
    X = (float*)malloc(N*sizeof(float));
    Y = (float*)malloc(N*sizeof(float));
    for (int i = 0; i<Nx; i++){
        for (int j = 0; j<Ny; j++){
            X[j*Nx + i] = i;
            Y[j*Nx + i] = j;
        }
    }

    std::cout << "1";
    // Parameters
    float *psi, *omega, *u, *v;
    psi = (float*)malloc(N*sizeof(float));
    omega = (float*)malloc(N*sizeof(float));
    u = (float*)malloc(N*sizeof(float));
    v = (float*)malloc(N*sizeof(float));

    std::cout << "2";
    // imposing boundary
    for (int i = 0; i<Nx; i++){
        for (int j = 0; j<Ny; j++){
            if (j == Ny-1){
                u[(Ny-1)*Nx + i] = lid;
            } else {
                u[j*Nx + i] = 0.0;
            }
        }
    }
    
    std::cout << "3";
    // Initialize the grid and block
	dim3 block(32, 32);
	dim3 grid((Ny + block.x - 1) / block.x, (Nx + block.y - 1) / block.y);

    std::cout << "4";
    // Declare parameters in cuda
    float *d_X, *d_Y, *d_psi, *d_omega, *d_u, *d_v,
        *d_psi_new, *d_psi_SOR; //temporary matrix
    cudaMalloc(&d_X, N*sizeof(float));
    cudaMalloc(&d_Y, N*sizeof(float));
    cudaMalloc(&d_psi, N*sizeof(float));
    cudaMalloc(&d_omega, N*sizeof(float));
    cudaMalloc(&d_u, N*sizeof(float));
    cudaMalloc(&d_v, N*sizeof(float));
    cudaMalloc(&d_psi_new, N*sizeof(float));
    cudaMalloc(&d_psi_SOR, N*sizeof(float));

    std::cout << "5";
    // copy the parameter values
    cudaMemcpy(d_X, X, N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u, N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemset(d_psi, 0, N*sizeof(float));
    cudaMemset(d_omega, 0, N*sizeof(float));
    cudaMemset(d_v, 0, N*sizeof(float));
    cudaMemset(d_psi_new, 0, N*sizeof(float));
    cudaMemset(d_psi_SOR, 0, N*sizeof(float));

    std::cout << "6";
    // Record the time
    float elapsedTime = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaSetDevice(0);
    cudaEventRecord(start,0);
    
    std::cout << "7";
    // GPU function
    // GPU_Navier <<<grid, block>>> 
    //             (dy, dx, dt, Nt, 
    //              Nmax_SOR_psi, psi_Toll, 
    //              lid, Re, 
    //              d_psi, d_omega, d_u, d_v,
    //              d_psi_new, d_psi_SOR);
                 
    for (int k = 0; k<Nt; k++){
        boundary <<<grid, block>>> (dy, dx, Nx, Ny, lid, d_psi, d_omega);
        vorticity <<<grid, block>>> (dy, dx, dt, Nx, Ny, Re, d_psi, d_omega, d_u, d_v, d_psi_new);
        stream <<<grid, block>>> (dy, dx, Nx, Ny, Nmax_SOR_psi, psi_Toll, d_psi, d_omega, d_psi_new, d_psi_SOR);
        velocity <<<grid, block>>> (dy, dx, Nx, Ny, d_psi, d_u, d_v, d_psi_new);
    }
    std::cout << "8";

    // Cuda synchronize and get time
    cudaDeviceSynchronize();
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // get the parameters
    cudaMemcpy(psi, d_psi, N*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(omega, d_omega, N*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(u, d_u, N*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(v, d_v, N*sizeof(float),cudaMemcpyDeviceToHost);

    // save the result in text
    std::ofstream psi_text("psi.txt");
    for (int i = 0; i<Nx; i++){
        for (int j = 0; j<Ny; j++){
            if ( j != 0){
                psi_text << ",";
            }
            psi_text << psi[j*Nx + i];
        }
        psi_text << "\n";
    }

    // Free the memory
    cudaFree(d_X); cudaFree(d_Y);
    cudaFree(d_psi); cudaFree(d_omega);
    cudaFree(d_u); cudaFree(d_v);
    cudaFree(d_psi_new); cudaFree(d_psi_SOR);
    free(X); free(Y);
    free(psi); free(omega);
    free(u); free(v);

    std::cout << "Done";
    return 0;
}