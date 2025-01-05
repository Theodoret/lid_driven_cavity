#include "header.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <math.h>

int main(int argc, char *argv[]){
    // Hyperparameters
    const int Nx = 51, Ny = 51, N = Nx*Ny, Nt = 10000;
    float lid = 1, Re = 1000,
        xmin = 0, xmax = 1, dx = (xmax - xmin)/(Nx - 1), 
        ymin = 0, ymax = 1, dy = (ymax - ymin)/(Ny - 1), 
        tmin = 0, tmax = 100, dt = (tmax - tmin)/Nt;
    float Nmax_SOR_psi = 200, psi_Toll = 0.01;

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
            // X[j*Nx + i] = i;
            // Y[j*Nx + i] = j;
            X[j*Nx + i] = x[i];
            Y[j*Nx + i] = y[j];
        }
    }

    // Parameters
    float *psi, *omega, *u, *v;
    psi = (float*)malloc(N*sizeof(float));
    omega = (float*)malloc(N*sizeof(float));
    u = (float*)malloc(N*sizeof(float));
    v = (float*)malloc(N*sizeof(float));

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
    
    // Initialize the grid and block
	dim3 block(32, 32);
	dim3 grid((Ny + block.x - 1) / block.x, (Nx + block.y - 1) / block.y);

    // Declare parameters in cuda
    float *d_psi, *d_omega, *d_u, *d_v,
        *d_psi_new, *d_psi_SOR, *d_psi_SORerr; //temporary matrix
    cudaMalloc(&d_psi, N*sizeof(float));
    cudaMalloc(&d_omega, N*sizeof(float));
    cudaMalloc(&d_u, N*sizeof(float));
    cudaMalloc(&d_v, N*sizeof(float));
    cudaMalloc(&d_psi_new, N*sizeof(float));
    cudaMalloc(&d_psi_SOR, N*sizeof(float));
    cudaMalloc(&d_psi_SORerr, sizeof(float));

    // copy the parameter values
    cudaMemcpy(d_u, u, N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemset(d_psi, 0, N*sizeof(float));
    cudaMemset(d_omega, 0, N*sizeof(float));
    cudaMemset(d_v, 0, N*sizeof(float));
    cudaMemset(d_psi_new, 0, N*sizeof(float));
    cudaMemset(d_psi_SOR, 0, N*sizeof(float));

    // Record the time
    float elapsedTime = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaSetDevice(0);
    cudaEventRecord(start,0);
    
    // GPU function
    for (int k = 0; k<Nt; k++){
        // boundary <<<grid, block>>> (dy, dx, Nx, Ny, lid, d_psi, d_omega);
        simple_boundary <<<grid, block>>> (dy, dx, Nx, Ny, lid, d_omega, d_u, d_v);
        cudaMemset(d_psi_SORerr, 0, sizeof(float));
        vorticity <<<grid, block>>> (dy, dx, dt, Nx, Ny, Re, d_psi, d_omega, d_u, d_v, d_psi_new);
        stream <<<grid, block>>> (dy, dx, Nx, Ny, Nmax_SOR_psi, psi_Toll, d_psi, d_omega, d_psi_new, d_psi_SOR, d_psi_SORerr);
        velocity <<<grid, block>>> (dy, dx, Nx, Ny, d_psi, d_u, d_v, d_psi_new);
    }

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
    write("X", X, Nx, Ny);
    write("Y", Y, Nx, Ny);
    write("psi", psi, Nx, Ny);
    write("omega", omega, Nx, Ny);
    write("u", u, Nx, Ny);
    write("v", v, Nx, Ny);

    // Free the memory
    cudaFree(d_psi); cudaFree(d_omega);
    cudaFree(d_u); cudaFree(d_v);
    cudaFree(d_psi_new); cudaFree(d_psi_SOR);
    cudaFree(d_psi_SORerr);
    free(X); free(Y);
    free(psi); free(omega);
    free(u); free(v);

    std::cout << "runtime: " << elapsedTime << " ms \n";
    return 0;
}