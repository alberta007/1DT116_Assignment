#ifndef AGENTS_SOA_H
#define AGENTS_SOA_H

#include <vector>
#include <stddef.h>

struct AgentsSoA {
    // Current positions (int)
    // Example: x[3] = 5, y[3] = 7 means agent 3 is at position (5, 7)
    // [5, 3, 5, 7, 3, 2, 1, 0, 0, 1] means 10 agents are at x positions 5, 3, 5, ...
    // [7, 2, 3, 7, 2, 1, 0, 1, 2, 2] means 10 agents are at y positions 7, 2, 3, ...
    std::vector<int> x;
    std::vector<int> y;

    // Which waypoint index each agent is currently heading toward
    // Example: if currentWaypointIndex[3] == 1, then agent 3 is heading toward waypoint 1.
    // [0, 1, 0, 2, 1, 0, 0, 1, 2, 2] means 10 agents are heading toward waypoints 0, 1, 0, 2, ...
    std::vector<int> currentWaypointIndex;

    // Resize all vectors to hold n elements.
    void resize(size_t n) {
        x.resize(n);
        y.resize(n);

        currentWaypointIndex.resize(n);
    }
};

#endif // AGENTS_SOA_H
