#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#define SizeX 51
#define SIzeY 51

__global__ void helloCUDA()
{
    for (int i = 0; i<10; i++){
        __syncthreads();
        printf("Hello, CUDA!\n");
    }
}

int main()
{
    helloCUDA<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}