// heatmap_cuda.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>

#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE (SIZE * CELLSIZE)
#define WEIGHTSUM 273

// Kernel 1: Fade out the heatmap (multiply each value by 0.80)
__global__ void fadeKernel(int* d_heatmap, int width, int height, float factor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        d_heatmap[idx] = (int)roundf(d_heatmap[idx] * factor);
    }
}

// Kernel 2: Increment the heatmap at agent-desired positions
__global__ void updateAgentsKernel(int* d_heatmap, int width, int height,
                                   const int* d_desiredXs, const int* d_desiredYs,
                                   int numAgents, int increment) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numAgents) {
        int x = d_desiredXs[i];
        int y = d_desiredYs[i];
        if (x >= 0 && x < width && y >= 0 && y < height) {
            int idx = y * width + x;
            atomicAdd(&d_heatmap[idx], increment);
        }
    }
}

// Kernel 3: Clamp heatmap values to a maximum (255)
__global__ void clampKernel(int* d_heatmap, int width, int height, int maxVal) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        if(d_heatmap[idx] > maxVal)
            d_heatmap[idx] = maxVal;
    }
}

// Kernel 4: Scale the heatmap to the size of the current visual representation
__global__ void scaleKernel(const int* d_heatmap, int* d_scaledHeatmap,
                            int origWidth, int origHeight, int scaleFactor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // x-coordinate in scaled image
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int scaledWidth = origWidth * scaleFactor;
    if (x < scaledWidth && y < (origHeight * scaleFactor)) {
        int origX = x / scaleFactor;
        int origY = y / scaleFactor;
        d_scaledHeatmap[y * scaledWidth + x] = d_heatmap[origY * origWidth + origX];
    }
}


// Kernel 5: Apply a Gaussian blur filter using shared memory.
// This kernel uses a dynamic shared memory tile that covers the block plus a 2-pixel halo on all sides.
__global__ void blurKernel(const int* d_scaledHeatmap, int* d_blurredHeatmap,
                           int width, int height) {
    // Calculate shared memory tile dimensions:
    // Each block loads blockDim.x * blockDim.y elements plus a halo of 2 pixels on every side.
    extern __shared__ int s_data[];
    int s_width = blockDim.x + 4; // add 2 on left and 2 on right
    // Compute indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * blockDim.x + tx;
    int gy = blockIdx.y * blockDim.y + ty;
    
    // Load the shared memory tile (with halo) using a cooperative loop.
    for (int j = ty; j < blockDim.y + 4; j += blockDim.y) {
        for (int i = tx; i < blockDim.x + 4; i += blockDim.x) {
            int global_x = blockIdx.x * blockDim.x + i - 2; // subtract halo offset
            int global_y = blockIdx.y * blockDim.y + j - 2;
            if (global_x >= 0 && global_x < width && global_y >= 0 && global_y < height)
                s_data[j * s_width + i] = d_scaledHeatmap[global_y * width + global_x];
            else
                s_data[j * s_width + i] = 0;
        }
    }
    __syncthreads();
    
    // Only process pixels where the full 5x5 window is available
    if (gx >= 2 && gx < width - 2 && gy >= 2 && gy < height - 2) {
        int sum = 0;
        int w[5][5] = {
            { 1,  4,  7,  4, 1 },
            { 4, 16, 26, 16, 4 },
            { 7, 26, 41, 26, 7 },
            { 4, 16, 26, 16, 4 },
            { 1,  4,  7,  4, 1 }
        };
        // In shared memory, the pixel corresponding to (gx,gy) is at (tx+2, ty+2)
        for (int ky = 0; ky < 5; ky++) {
            for (int kx = 0; kx < 5; kx++) {
                int s_val = s_data[(ty + ky) * s_width + (tx + kx)];
                sum += w[ky][kx] * s_val;
            }
        }
        int value = sum / WEIGHTSUM;
        // Pack the output into ARGB where red is fixed (0x00FF0000) and the alpha channel represents the heat.
        d_blurredHeatmap[gy * width + gx] = 0x00FF0000 | (value << 24);
        
    }
}

// Host function that wraps all the steps.
// h_heatmap, h_scaledHeatmap, and h_blurredHeatmap point to contiguous host memory blocks.
// h_desiredXs and h_desiredYs are arrays holding agent desired positions (each of length numAgents).
extern "C" void updateHeatmapCuda(int* h_heatmap, int* h_scaledHeatmap, int* h_blurredHeatmap,
                                  int numAgents, int* h_desiredXs, int* h_desiredYs)
{
    // Device pointers
    int *d_heatmap, *d_scaledHeatmap, *d_blurredHeatmap;
    int *d_desiredXs, *d_desiredYs;
    
    size_t heatmapBytes = SIZE * SIZE * sizeof(int);
    size_t scaledBytes  = SCALED_SIZE * SCALED_SIZE * sizeof(int);
    size_t agentsBytes  = numAgents * sizeof(int);
    
    // Allocate device memory
    cudaMalloc((void**)&d_heatmap, heatmapBytes);
    cudaMalloc((void**)&d_scaledHeatmap, scaledBytes);
    cudaMalloc((void**)&d_blurredHeatmap, scaledBytes);
    cudaMalloc((void**)&d_desiredXs, agentsBytes);
    cudaMalloc((void**)&d_desiredYs, agentsBytes);
    
    // Copy host data to device
    cudaMemcpy(d_heatmap, h_heatmap, heatmapBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_desiredXs, h_desiredXs, agentsBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_desiredYs, h_desiredYs, agentsBytes, cudaMemcpyHostToDevice);
    
    // Set up execution configuration for the heatmap (SIZE x SIZE)
    dim3 blockDim2D(16, 16);
    dim3 gridDim2D((SIZE + blockDim2D.x - 1) / blockDim2D.x, (SIZE + blockDim2D.y - 1) / blockDim2D.y);
    
    // Step 1: Fade out the heatmap
    fadeKernel<<<gridDim2D, blockDim2D>>>(d_heatmap, SIZE, SIZE, 0.80f);
    
    // Step 2: Increment heat for each agent's desired position (use 1D configuration)
    int threadsPerBlock = 256;
    int blocksAgents = (numAgents + threadsPerBlock - 1) / threadsPerBlock;
    updateAgentsKernel<<<blocksAgents, threadsPerBlock>>>(d_heatmap, SIZE, SIZE, d_desiredXs, d_desiredYs, numAgents, 40);
    
    // Step 3: Clamp heatmap values to a maximum of 255
    clampKernel<<<gridDim2D, blockDim2D>>>(d_heatmap, SIZE, SIZE, 255);
    
    // Step 4: Scale the heatmap to the size used for display
    dim3 blockDimScale(16, 16);
    dim3 gridDimScale((SCALED_SIZE + blockDimScale.x - 1) / blockDimScale.x, (SCALED_SIZE + blockDimScale.y - 1) / blockDimScale.y);
    scaleKernel<<<gridDimScale, blockDimScale>>>(d_heatmap, d_scaledHeatmap, SIZE, SIZE, CELLSIZE);
    
    // Step 5: Apply the Gaussian blur filter.
    // Choose a block size (e.g., 16x16) and calculate the required shared memory size.
    dim3 blockDimBlur(16, 16);
    dim3 gridDimBlur((SCALED_SIZE + blockDimBlur.x - 1) / blockDimBlur.x, (SCALED_SIZE + blockDimBlur.y - 1) / blockDimBlur.y);
    size_t sharedMemSize = (blockDimBlur.x + 4) * (blockDimBlur.y + 4) * sizeof(int);
    blurKernel<<<gridDimBlur, blockDimBlur, sharedMemSize>>>(d_scaledHeatmap, d_blurredHeatmap, SCALED_SIZE, SCALED_SIZE);

    
    // Copy the final blurred heatmap back to host memory.
    cudaMemcpy(h_blurredHeatmap, d_blurredHeatmap, scaledBytes, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_heatmap);
    cudaFree(d_scaledHeatmap);
    // cudaFree(d_blurredHeatmap);
    cudaFree(d_desiredXs);
    cudaFree(d_desiredYs);
}
