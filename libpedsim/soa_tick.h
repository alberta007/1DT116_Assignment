// tick_soa.h
#ifndef TICK_SOA_H
#define TICK_SOA_H

#include "soa_agent.h"
#include "waypoints_soa.h"

// Performs one simulation tick on the agents.
// Computes a new position by moving from the current position toward the destination.
void tickSoA(AgentsSoA &agents, const WaypointsSoA &waypoints);

#endif // TICK_SOA_H
