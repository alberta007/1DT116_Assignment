// tick_cuda.cu
#ifndef NOCUDA
#include <cuda_runtime.h>
#include <math.h>
#include "tick_cuda.h"  // Declaration header (if needed)
#include <cstdio>
#endif

// CUDA Kernel: Each thread handles one agent.
__global__ void tickCuda_kernel(
    int *agent_x, int *agent_y,
    float *destX, float *destY, float *destR,
    int *currentWaypointIndex,
    const float *waypoint_x, const float *waypoint_y, const float *waypoint_r,
    int numAgents, int numWaypoints)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numAgents) return;

    // Load current agent position.
    float ax = (float)agent_x[idx];
    float ay = (float)agent_y[idx];

    // Load current destination from the agent's SoA.
    float wx = destX[idx];
    float wy = destY[idx];
    float wr = destR[idx];

    // Load the current waypoint index.
    int wpIdx = currentWaypointIndex[idx];

    // Compute squared distance to destination.
    float dx = wx - ax;
    float dy = wy - ay;
    float dist2 = dx * dx + dy * dy;
    float wr2 = wr * wr;

    // If the agent is within the destination radius, update its waypoint.
    if (dist2 < wr2) {
        wpIdx = (wpIdx + 1) % numWaypoints;
        currentWaypointIndex[idx] = wpIdx;
        // Update destination values from global waypoint data.
        wx = waypoint_x[wpIdx];
        wy = waypoint_y[wpIdx];
        wr = waypoint_r[wpIdx];
        destX[idx] = wx;
        destY[idx] = wy;
        destR[idx] = wr;
        // Recompute differences for the movement step.
        dx = wx - ax;
        dy = wy - ay;
        dist2 = dx * dx + dy * dy;
    }

    // Compute Euclidean distance.
    float len = sqrtf(dist2);
    if (len < 1e-6f) len = 1.0f;  // Avoid division by zero.

    // Compute unit vector toward the destination.
    float nx = dx / len;
    float ny = dy / len;

    // Update position by one unit step.
    ax += nx;
    ay += ny;
    agent_x[idx] = (int)roundf(ax);
    agent_y[idx] = (int)roundf(ay); 
}

// Host function: Launches the CUDA kernel.
void tickCuda(AgentsSoA &agents, const WaypointsSoA &waypoints) {
#ifndef NOCUDA
    float *d_agents_destX, *d_agents_destY, *d_agents_destR;
    int *d_agents_currentWaypointIndex, *d_agents_x, *d_agents_y;
    float *d_waypoints_x, *d_waypoints_y, *d_waypoints_r;
    
    int numAgents = agents.x.size();
    int numWaypoints = waypoints.x.size();
    
    cudaMalloc((void**)&d_agents_x, numAgents * sizeof(int));
    cudaMalloc((void**)&d_agents_y, numAgents * sizeof(int));
    cudaMalloc((void**)&d_agents_destX, numAgents * sizeof(float));
    cudaMalloc((void**)&d_agents_destY, numAgents * sizeof(float));
    cudaMalloc((void**)&d_agents_destR, numAgents * sizeof(float));
    cudaMalloc((void**)&d_agents_currentWaypointIndex, numAgents * sizeof(int));

    cudaMalloc((void**)&d_waypoints_x, numWaypoints * sizeof(float));
    cudaMalloc((void**)&d_waypoints_y, numWaypoints * sizeof(float));
    cudaMalloc((void**)&d_waypoints_r, numWaypoints * sizeof(float));

    cudaMemcpy(d_agents_x, agents.x.data(), numAgents * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_agents_y, agents.y.data(), numAgents * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_agents_destX, agents.destX.data(), numAgents * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_agents_destY, agents.destY.data(), numAgents * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_agents_destR, agents.destR.data(), numAgents * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_agents_currentWaypointIndex, agents.currentWaypointIndex.data(), numAgents * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_waypoints_x, waypoints.x.data(), numWaypoints * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_waypoints_y, waypoints.y.data(), numWaypoints * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_waypoints_r, waypoints.r.data(), numWaypoints * sizeof(float), cudaMemcpyHostToDevice);

    // Choose a block size and calculate grid size.
    int blockSize = 256;
    int gridSize = (numAgents + blockSize - 1) / blockSize;

    // Launch the kernel.
    tickCuda_kernel<<<gridSize, blockSize>>>(
        d_agents_x, d_agents_y,
        d_agents_destX, d_agents_destY, d_agents_destR,
        d_agents_currentWaypointIndex,
        d_waypoints_x, d_waypoints_y, d_waypoints_r,
        numAgents, numWaypoints
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Check for execution errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error after synchronization: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(agents.x.data(), d_agents_x, numAgents * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(agents.y.data(), d_agents_y, numAgents * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(agents.destX.data(), d_agents_destX, numAgents * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(agents.destY.data(), d_agents_destY, numAgents * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(agents.destR.data(), d_agents_destR, numAgents * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(agents.currentWaypointIndex.data(), d_agents_currentWaypointIndex, numAgents * sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy((float *)waypoints.x.data(), d_waypoints_x, numWaypoints * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy((float *)waypoints.y.data(), d_waypoints_y, numWaypoints * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy((float *)waypoints.r.data(), d_waypoints_r, numWaypoints * sizeof(float), cudaMemcpyDeviceToHost);

    // Synchronize the device to ensure the kernel has finished.
    cudaFree(d_agents_x);
    cudaFree(d_agents_y);
    cudaFree(d_agents_destX);
    cudaFree(d_agents_destY);
    cudaFree(d_agents_destR);
    cudaFree(d_agents_currentWaypointIndex);
    cudaFree(d_waypoints_x);
    cudaFree(d_waypoints_y);
    cudaFree(d_waypoints_r);
#endif
}
