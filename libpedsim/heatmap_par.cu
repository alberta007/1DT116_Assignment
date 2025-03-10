#ifndef NOCUDA
#include "heatmap_par.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE (SIZE * CELLSIZE)
#define WEIGHTSUM 273

__constant__ int w[5][5] = {
    { 1,  4,  7,  4, 1 },
    { 4, 16, 26, 16, 4 },
    { 7, 26, 41, 26, 7 },
    { 4, 16, 26, 16, 4 },
    { 1,  4,  7,  4, 1 }
};

__global__ void updateHeatKernel(int *heatmap, int numAgents, int *d_agents_desX, int *d_agents_desY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= SIZE || y >= SIZE) return;
    
    // Apply heat decay
    int idx = y * SIZE + x;
    heatmap[idx] = (int)roundf(heatmap[idx] * 0.8f);
}

__global__ void addAgentHeatKernel(int *heatmap, int numAgents, int *d_agents_desX, int *d_agents_desY) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= numAgents) return;
    
    int x = d_agents_desX[idx];
    int y = d_agents_desY[idx];
    
    if (x >= 0 && x < SIZE && y >= 0 && y < SIZE) {
        atomicAdd(&heatmap[y * SIZE + x], 40);
    }
}

__global__ void clampHeatKernel(int *heatmap) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= SIZE || y >= SIZE) return;
    
    int idx = y * SIZE + x;
    if (heatmap[idx] > 255) {
        heatmap[idx] = 255;
    }
}

__global__ void scaleHeatmapKernel(int* heatmap, int* scaled_heatmap) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= SIZE || y >= SIZE) return;
    
    int value = heatmap[y * SIZE + x];
    
    // Each thread scales its cell to CELLSIZE×CELLSIZE cells in the scaled heatmap
    for (int cy = 0; cy < CELLSIZE; cy++) {
        for (int cx = 0; cx < CELLSIZE; cx++) {
            scaled_heatmap[(y * CELLSIZE + cy) * SCALED_SIZE + (x * CELLSIZE + cx)] = value;
        }
    }
}

__global__ void blurHeatmapKernel(int *scaled_heatmap, int *blurred_heatmap) {
    // Shared memory for efficient blurring
    extern __shared__ int s_data[];
    int s_width = blockDim.x + 4;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * blockDim.x + tx;
    int gy = blockIdx.y * blockDim.y + ty;
    
    // Load data into shared memory (including halo region)
    for (int j = ty; j < blockDim.y + 4; j += blockDim.y) {
        for (int i = tx; i < blockDim.x + 4; i += blockDim.x) {
            int global_x = blockIdx.x * blockDim.x + i - 2; // subtract halo offset
            int global_y = blockIdx.y * blockDim.y + j - 2;
            
            if (global_x >= 0 && global_x < SCALED_SIZE && global_y >= 0 && global_y < SCALED_SIZE) {
                s_data[j * s_width + i] = scaled_heatmap[global_y * SCALED_SIZE + global_x];
            } else {
                s_data[j * s_width + i] = 0;
            }
        }
    }
    
    __syncthreads();
    
    // Apply Gaussian blur filter
    if (gx >= 2 && gx < SCALED_SIZE - 2 && gy >= 2 && gy < SCALED_SIZE - 2) {
        int sum = 0;
        
        for (int ky = 0; ky < 5; ky++) {
            for (int kx = 0; kx < 5; kx++) {
                int s_val = s_data[(ty + ky) * s_width + (tx + kx)];
                sum += w[ky][kx] * s_val;
            }
        }
        
        int value = sum / WEIGHTSUM;
        blurred_heatmap[gy * SCALED_SIZE + gx] = 0x00FF0000 | (value << 24);
    }
}

// Host function that updates the heatmap
void updateHeatMapCuda(int *heatmap, int *scaled_heatmap, int *blurred_heatmap, 
                       int *desiredXs, int *desiredYs, int numAgents) {
#ifndef NOCUDA
    // Allocate device memory
    int *d_heatmap, *d_scaled_heatmap, *d_blurred_heatmap;
    int *d_agents_desX, *d_agents_desY;
    
    size_t heatmapBytes = SIZE * SIZE * sizeof(int);
    size_t scaledBytes = SCALED_SIZE * SCALED_SIZE * sizeof(int);
    size_t agentsBytes = numAgents * sizeof(int);
    
    cudaMalloc((void**)&d_heatmap, heatmapBytes);
    cudaMalloc((void**)&d_scaled_heatmap, scaledBytes);
    cudaMalloc((void**)&d_blurred_heatmap, scaledBytes);
    cudaMalloc((void**)&d_agents_desX, agentsBytes);
    cudaMalloc((void**)&d_agents_desY, agentsBytes);
    
    // Copy host memory to device
    cudaMemcpy(d_heatmap, heatmap, heatmapBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_agents_desX, desiredXs, agentsBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_agents_desY, desiredYs, agentsBytes, cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions for the heat update kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((SIZE + blockDim.x - 1) / blockDim.x, 
                 (SIZE + blockDim.y - 1) / blockDim.y);
    
    // Launch heat update kernel (decay)
    updateHeatKernel<<<gridDim, blockDim>>>(d_heatmap, numAgents, d_agents_desX, d_agents_desY);
    
    // Launch kernel to add agent heat
    int agentBlockSize = 256;
    int agentGridSize = (numAgents + agentBlockSize - 1) / agentBlockSize;
    addAgentHeatKernel<<<agentGridSize, agentBlockSize>>>(d_heatmap, numAgents, d_agents_desX, d_agents_desY);
    
    // Launch kernel to clamp heat values
    clampHeatKernel<<<gridDim, blockDim>>>(d_heatmap);
    
    // Launch scaling kernel
    scaleHeatmapKernel<<<gridDim, blockDim>>>(d_heatmap, d_scaled_heatmap);
    
    // Define grid and block dimensions for the blur kernel
    dim3 blurBlockDim(16, 16);
    dim3 blurGridDim((SCALED_SIZE + blurBlockDim.x - 1) / blurBlockDim.x, 
                     (SCALED_SIZE + blurBlockDim.y - 1) / blurBlockDim.y);
    
    // Calculate shared memory size
    size_t sharedMemSize = (blurBlockDim.x + 4) * (blurBlockDim.y + 4) * sizeof(int);
    
    // Launch blur kernel with shared memory
    blurHeatmapKernel<<<blurGridDim, blurBlockDim, sharedMemSize>>>(d_scaled_heatmap, d_blurred_heatmap);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    // Synchronize
    cudaDeviceSynchronize();
    
    // Copy results back to host
    cudaMemcpy(heatmap, d_heatmap, heatmapBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(scaled_heatmap, d_scaled_heatmap, scaledBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(blurred_heatmap, d_blurred_heatmap, scaledBytes, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_heatmap);
    cudaFree(d_scaled_heatmap);
    cudaFree(d_blurred_heatmap);
    cudaFree(d_agents_desX);
    cudaFree(d_agents_desY);
#endif
}
#endif