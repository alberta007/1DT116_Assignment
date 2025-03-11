#ifndef NOCUDA
#include "heatmap_par.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>


// WEIGHTSUM: the normalization factor for the Gaussian blur.
#define WEIGHTSUM 273

// Device pointers allocated once and used across ticks.
static int *d_heatmap = nullptr;
static int *d_scaled_heatmap = nullptr;
static int *d_blurred_heatmap = nullptr;
static int *d_agents_desX = nullptr;
static int *d_agents_desY = nullptr;

// Constant memory for the Gaussian kernel weights (5x5 matrix).
// Using constant memory makes these values fast to access in all threads.
__constant__ int w[5][5] = {
    { 1,  4,  7,  4, 1 },
    { 4, 16, 26, 16, 4 },
    { 7, 26, 41, 26, 7 },
    { 4, 16, 26, 16, 4 },
    { 1,  4,  7,  4, 1 }
};


// Setup function: Allocates device memory once.
void setupCudaHeatmap(int numAgents) {
    size_t heatmapBytes = SIZE * SIZE * sizeof(int);
    size_t scaledBytes  = SCALED_SIZE * SCALED_SIZE * sizeof(int);
    size_t agentsBytes  = numAgents * sizeof(int);
    
    cudaMalloc((void**)&d_heatmap, heatmapBytes);
    cudaMalloc((void**)&d_scaled_heatmap, scaledBytes);
    cudaMalloc((void**)&d_blurred_heatmap, scaledBytes);
    cudaMalloc((void**)&d_agents_desX, agentsBytes);
    cudaMalloc((void**)&d_agents_desY, agentsBytes);
}

// -----------------------------------------------------------------
// Cleanup function: Frees the device memory allocated during setup.
void cleanupCudaHeatmap() {
    cudaFree(d_heatmap);
    cudaFree(d_scaled_heatmap);
    cudaFree(d_blurred_heatmap);
    cudaFree(d_agents_desX);
    cudaFree(d_agents_desY);
    
    d_heatmap = nullptr;
    d_scaled_heatmap = nullptr;
    d_blurred_heatmap = nullptr;
    d_agents_desX = nullptr;
    d_agents_desY = nullptr;
}



// -----------------------------------------------------------------
// Kernel: updateHeatKernel
// This kernel applies a decay factor to each heatmap cell (fading effect).
// Each thread processes one pixel in the heatmap.
__global__ void updateHeatKernel(int *heatmap, int numAgents, int *d_agents_desX, int *d_agents_desY) {
    // Compute the x and y coordinate for this thread.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Ensure thread is within heatmap bounds.
    if (x >= SIZE || y >= SIZE) return;
    
    // Compute the linear index for the heatmap array.
    int idx = y * SIZE + x;
    // Apply heat decay: multiply the current value by 0.8 and round.
    heatmap[idx] = (int)roundf(heatmap[idx] * 0.8f);
}

// -----------------------------------------------------------------
// Kernel: addAgentHeatKernel
// This kernel increments heatmap values at positions desired by agents.
// Each thread processes one agent; atomicAdd is used to prevent data races.
__global__ void addAgentHeatKernel(int *heatmap, int numAgents, int *d_agents_desX, int *d_agents_desY) {
    // Compute a global index for each agent.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Ensure the thread corresponds to a valid agent.
    if (idx >= numAgents) return;
    
    // Retrieve the desired x and y position for this agent.
    int x = d_agents_desX[idx];
    int y = d_agents_desY[idx];
    
    // Check that the agent's desired position is within the heatmap bounds.
    if (x >= 0 && x < SIZE && y >= 0 && y < SIZE) {
        // Atomically add 40 to the corresponding heatmap cell.
        atomicAdd(&heatmap[y * SIZE + x], 40);
    }
}

// -----------------------------------------------------------------
// Kernel: clampHeatKernel
// This kernel clamps the heatmap values so that they do not exceed 255.
__global__ void clampHeatKernel(int *heatmap) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= SIZE || y >= SIZE) return;
    
    int idx = y * SIZE + x;
    // Clamp the value to 255 if it exceeds that.
    if (heatmap[idx] > 255) {
        heatmap[idx] = 255;
    }
}

// -----------------------------------------------------------------
// Kernel: scaleHeatmapKernel
// This kernel scales the heatmap up for visualization by copying each cell's value
// into a CELLSIZE x CELLSIZE block in the scaled heatmap.
__global__ void scaleHeatmapKernel(int* heatmap, int* scaled_heatmap) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Make sure we process only valid cells from the original heatmap.
    if (x >= SIZE || y >= SIZE) return;
    
    // Get the heat value at this cell.
    int value = heatmap[y * SIZE + x];
    
    // Each original heatmap cell expands to a CELLSIZE×CELLSIZE block.
    for (int cy = 0; cy < CELLSIZE; cy++) {
        for (int cx = 0; cx < CELLSIZE; cx++) {
            // Compute the index in the scaled heatmap and copy the value.
            scaled_heatmap[(y * CELLSIZE + cy) * SCALED_SIZE + (x * CELLSIZE + cx)] = value;
        }
    }
}

// -----------------------------------------------------------------
// Kernel: blurHeatmapKernel
// This kernel applies a Gaussian blur to the scaled heatmap using shared memory.
// A tile (plus a 2-pixel halo) is loaded into shared memory to reduce global memory accesses.
__global__ void blurHeatmapKernel(int *scaled_heatmap, int *blurred_heatmap) {
    // Declare shared memory for a tile with a 2-pixel halo on all sides.
    extern __shared__ int s_data[];
    // The shared memory width equals the block width plus 4 (2 pixels on each side).
    int s_width = blockDim.x + 4;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // Compute global coordinates in the scaled heatmap.
    int gx = blockIdx.x * blockDim.x + tx;
    int gy = blockIdx.y * blockDim.y + ty;
    
    // Load the data into shared memory including the halo.
    // Loop to cover the entire tile plus halo if block dimensions are smaller.
    for (int j = ty; j < blockDim.y + 4; j += blockDim.y) {
        for (int i = tx; i < blockDim.x + 4; i += blockDim.x) {
            // Compute global coordinates with halo offset (subtract 2).
            int global_x = blockIdx.x * blockDim.x + i - 2;
            int global_y = blockIdx.y * blockDim.y + j - 2;
            
            // Load from global memory if within bounds; otherwise, set to 0.
            if (global_x >= 0 && global_x < SCALED_SIZE && global_y >= 0 && global_y < SCALED_SIZE) {
                s_data[j * s_width + i] = scaled_heatmap[global_y * SCALED_SIZE + global_x];
            } else {
                s_data[j * s_width + i] = 0;
            }
        }
    }
    
    // Ensure all threads have loaded their portion of the shared memory.
    __syncthreads();
    
    // Only perform the blur if the full 5x5 window is available.
    if (gx >= 2 && gx < SCALED_SIZE - 2 && gy >= 2 && gy < SCALED_SIZE - 2) {
        int sum = 0;
        // Loop over the 5x5 window.
        for (int ky = 0; ky < 5; ky++) {
            for (int kx = 0; kx < 5; kx++) {
                // Access the pixel in shared memory corresponding to the current kernel position.
                int s_val = s_data[(ty + ky) * s_width + (tx + kx)];
                // Multiply by the Gaussian weight (from constant memory) and add to the sum.
                sum += w[ky][kx] * s_val;
            }
        }
        
        // Normalize the blurred value.
        int value = sum / WEIGHTSUM;
        // Pack the value into ARGB format: fixed red (0xFF) and alpha from the blurred value.
        blurred_heatmap[gy * SCALED_SIZE + gx] = 0x00FF0000 | (value << 24);
    }
}

// -----------------------------------------------------------------
// Host function: updateHeatMapCuda
// This function orchestrates the heatmap update on the GPU.
// It allocates device memory, copies host data to the device, launches kernels in sequence,
// synchronizes and copies results back to host memory, and then frees device memory.
void updateHeatMapCuda(int *heatmap, int *scaled_heatmap, int *blurred_heatmap, 
                                  int *desiredXs, int *desiredYs, int numAgents) {
    size_t heatmapBytes = SIZE * SIZE * sizeof(int);
    size_t scaledBytes  = SCALED_SIZE * SCALED_SIZE * sizeof(int);
    size_t agentsBytes  = numAgents * sizeof(int);
    
    // Copy host heatmap and agent position data to device.
    cudaMemcpy(d_heatmap, heatmap, heatmapBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_agents_desX, desiredXs, agentsBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_agents_desY, desiredYs, agentsBytes, cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions for operations on the heatmap.
    dim3 blockDim(16, 16);
    dim3 gridDim((SIZE + blockDim.x - 1) / blockDim.x, 
                 (SIZE + blockDim.y - 1) / blockDim.y);
    
    // Launch kernels in sequence.
    updateHeatKernel<<<gridDim, blockDim>>>(d_heatmap, numAgents, d_agents_desX, d_agents_desY);
    
    int agentBlockSize = 256;
    int agentGridSize = (numAgents + agentBlockSize - 1) / agentBlockSize;
    addAgentHeatKernel<<<agentGridSize, agentBlockSize>>>(d_heatmap, numAgents, d_agents_desX, d_agents_desY);
    
    clampHeatKernel<<<gridDim, blockDim>>>(d_heatmap);
    
    scaleHeatmapKernel<<<gridDim, blockDim>>>(d_heatmap, d_scaled_heatmap);
    
    // Define grid and block dimensions for the blur kernel.
    dim3 blurBlockDim(16, 16);
    dim3 blurGridDim((SCALED_SIZE + blurBlockDim.x - 1) / blurBlockDim.x, 
                     (SCALED_SIZE + blurBlockDim.y - 1) / blurBlockDim.y);
    size_t sharedMemSize = (blurBlockDim.x + 4) * (blurBlockDim.y + 4) * sizeof(int);
    blurHeatmapKernel<<<blurGridDim, blurBlockDim, sharedMemSize>>>(d_scaled_heatmap, d_blurred_heatmap);
    
    // Check for kernel launch errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    // Synchronize to ensure all kernels complete.
    cudaDeviceSynchronize();
    
    // Copy the results back to host memory.
    cudaMemcpy(heatmap, d_heatmap, heatmapBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(scaled_heatmap, d_scaled_heatmap, scaledBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(blurred_heatmap, d_blurred_heatmap, scaledBytes, cudaMemcpyDeviceToHost);
}
#endif
