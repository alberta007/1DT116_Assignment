// tick_cuda.h
#ifndef TICK_CUDA_H
#define TICK_CUDA_H

#include "soa_agent.h"
#include "waypoints_soa.h"

// Host function that launches the CUDA kernel.
void tickCuda(AgentsSoA &agents, const WaypointsSoA &waypoints);

#endif // TICK_CUDA_H
