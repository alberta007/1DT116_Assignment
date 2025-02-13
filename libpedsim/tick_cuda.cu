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
    int numAgents = agents.x.size();
    int numWaypoints = waypoints.x.size();

    // Choose a block size and calculate grid size.
    int blockSize = 256;
    int gridSize = (numAgents + blockSize - 1) / blockSize;

    // Launch the kernel.
    tickCuda_kernel<<<gridSize, blockSize>>>(
        agents.x.data(), agents.y.data(),
        agents.destX.data(), agents.destY.data(), agents.destR.data(),
        agents.currentWaypointIndex.data(),
        waypoints.x.data(), waypoints.y.data(), waypoints.r.data(),
        numAgents, numWaypoints
    );

    // Synchronize the device to ensure the kernel has finished.
    cudaDeviceSynchronize();
#endif
}
