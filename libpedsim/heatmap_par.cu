#ifndef NOCUDA

#include "heatmap_par.h"
#include <cuda_runtime.h>
#include <math.h>
#include <cstdio>

#endif

#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE (SIZE * CELLSIZE)
#define WEIGHTSUM 273


__global__ void updateHeatmapPar(int *heatmap, int *scaled_heatmap, int *blurred_heatmap, int sizeOfAgents, int *d_agents_desX, int *d_agents_desY)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= SIZE || y >= SIZE) return;

    if (x < SIZE && y < SIZE) {
        int idx = y * SIZE + x;
        heatmap[idx] = (int)roundf(heatmap[idx] * 0.8);
    }
	
    if (x < sizeOfAgents) {
        int x1 = d_agents_desX[x];
        int y1 = d_agents_desY[x];
        if (x1 >= 0 && x1 < SIZE && y1 >= 0 && y1 < SIZE) {
            int idx = y1 * SIZE + x1;
            atomicAdd(&heatmap[idx], 40);
        }
    }


    int idx = y * SIZE + x;
    if(heatmap[idx] > 255) {
        heatmap[idx] = 255;
    }

    int scaledSIZE = SIZE * CELLSIZE;
    if (x < scaledSIZE && y < scaledSIZE) {
        int origX = x / CELLSIZE;
        int origY = y / CELLSIZE;
        scaled_heatmap[y * scaledSIZE + x] = heatmap[origY * SIZE + origX];
    }
	
    extern __shared__ int s_data[];
    int s_SIZE = blockDim.x + 4; 
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * blockDim.x + tx;
    int gy = blockIdx.y * blockDim.y + ty;
    
    for (int j = ty; j < blockDim.y + 4; j += blockDim.y) {
        for (int i = tx; i < blockDim.x + 4; i += blockDim.x) {
            int global_x = blockIdx.x * blockDim.x + i - 2; // subtract halo offset
            int global_y = blockIdx.y * blockDim.y + j - 2;
            if (global_x >= 0 && global_x < SCALED_SIZE && global_y >= 0 && global_y < SCALED_SIZE)
                s_data[j * s_SIZE + i] = scaled_heatmap[global_y * SCALED_SIZE + global_x];
            else
                s_data[j * s_SIZE + i] = 0;
        }
    }
    __syncthreads();
    if (gx >= 2 && gx < SCALED_SIZE - 2 && gy >= 2 && gy < SCALED_SIZE - 2) {
        int sum = 0;
        int w[5][5] = {
            { 1,  4,  7,  4, 1 },
            { 4, 16, 26, 16, 4 },
            { 7, 26, 41, 26, 7 },
            { 4, 16, 26, 16, 4 },
            { 1,  4,  7,  4, 1 }
        };
        for (int ky = 0; ky < 5; ky++) {
            for (int kx = 0; kx < 5; kx++) {
                int s_val = s_data[(ty + ky) * s_SIZE + (tx + kx)];
                sum += w[ky][kx] * s_val;
            }
        }
        int value = sum / WEIGHTSUM;

        blurred_heatmap[gy * SCALED_SIZE + gx] = 0x00FF0000 | (value << 24);        
    }
}

// Updates the heatmap according to the agent positions
__host__ void updateHeatMapCuda(int *heatmap, int *scaled_heatmap, int *blurred_heatmap, int *desiredXs, int *desiredYs, int numAgents) {
#ifndef NOCUDA
    int *d_heatmap, *d_scaled_heatmap, *d_blurred_heatmap;
    int *d_agents_desX, *d_agents_desY;

    size_t heatmapBytes = SIZE * SIZE * sizeof(int);
    size_t scaledBytes  = SCALED_SIZE * SCALED_SIZE * sizeof(int);
    size_t agentsBytes  = numAgents * sizeof(int);
   

    cudaMalloc((void**)&d_heatmap, heatmapBytes);
    cudaMalloc((void**)&d_scaled_heatmap, scaledBytes);
    cudaMalloc((void**)&d_blurred_heatmap, scaledBytes);
    cudaMalloc((void**)&d_agents_desX, agentsBytes);
    cudaMalloc((void**)&d_agents_desY, agentsBytes);

    cudaMemcpy(d_blurred_heatmap, blurred_heatmap, scaledBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scaled_heatmap, scaled_heatmap, scaledBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_heatmap, heatmap, heatmapBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_agents_desX, desiredXs, agentsBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_agents_desY, desiredYs, agentsBytes, cudaMemcpyHostToDevice);


    dim3 blockDim(16, 16);
    dim3 gridDim((SIZE + blockDim.x - 1) / blockDim.x, (SIZE + blockDim.y - 1) / blockDim.y);
    size_t sharedMemSize = (blockDim.x + 4) * (blockDim.y + 4) * sizeof(int);

    updateHeatmapPar<<<gridDim, blockDim, sharedMemSize>>>(d_heatmap, d_scaled_heatmap, d_blurred_heatmap, numAgents, d_agents_desX, d_agents_desY);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();


    cudaMemcpy(heatmap, d_heatmap, heatmapBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(scaled_heatmap, d_scaled_heatmap, scaledBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(blurred_heatmap, d_blurred_heatmap, scaledBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_heatmap);
    cudaFree(d_scaled_heatmap);
    cudaFree(d_blurred_heatmap);
    cudaFree(d_agents_desX);
    cudaFree(d_agents_desY);

#endif
}