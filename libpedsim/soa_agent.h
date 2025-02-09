#ifndef AGENTS_SOA_H
#define AGENTS_SOA_H

#include <vector>
#include <stddef.h>

/**
 * Structure of Arrays (SoA) representation of agents.
 * We store the x, y, destX, destY, destR, and currentWaypointIndex of each agent in separate arrays.
 * This allows for better memory access patterns when computing the next desired position of an agent.
 * Thanks to SoA, we can compute the next desired position of all agents in parallel.
 */

struct AgentsSoA {
    // Current positions (int)
    // Example: x[3] = 5, y[3] = 7 means agent 3 is at position (5, 7)
    // [5, 3, 5, 7, 3, 2, 1, 0, 0, 1] means 10 agents are at x positions 5, 3, 5, ...
    // [7, 2, 3, 7, 2, 1, 0, 1, 2, 2] means 10 agents are at y positions 7, 2, 3, ...
    std::vector<int> x;
    std::vector<int> y;

    // Destination positions (int)
    // Example: destX[3] = 5, destY[3] = 7 means agent 3 is heading toward position (5, 7)
    // [5, 3, 5, 7, 3, 2, 1, 0, 0, 1] means 10 agents are heading toward x positions 5, 3, 5, ...
    // [7, 2, 3, 7, 2, 1, 0, 1, 2, 2] means 10 agents are heading toward y positions 7, 2, 3, ...
    // destR is the radius of the destination meaning the agent has arrived when it is within this radius
    std::vector<float> destX;
    std::vector<float> destY;
    std::vector<float> destR;

    // Which waypoint index each agent is currently heading toward
    // Example: if currentWaypointIndex[3] == 1, then agent 3 is heading toward waypoint 1.
    // [0, 1, 0, 2, 1, 0, 0, 1, 2, 2] means 10 agents are heading toward waypoints 0, 1, 0, 2, ...
    std::vector<int> currentWaypointIndex;

    // Resize all vectors to hold n elements.
    void resize(size_t n) {
        x.resize(n);
        y.resize(n);
        destX.resize(n);
        destY.resize(n);
        destR.resize(n);
        currentWaypointIndex.resize(n);
    }
};

#endif // AGENTS_SOA_H
