#ifndef AGENTS_SOA_H
#define AGENTS_SOA_H

#include <vector>
#include <stddef.h>

struct AgentsSoA {
    // Current positions (integers)
    std::vector<int> x;
    std::vector<int> y;

    // Destination coordinates (floats)
    std::vector<float> destX;
    std::vector<float> destY;

    // Computed desired positions (integers)
    std::vector<int> desiredX;
    std::vector<int> desiredY;

    // Which waypoint index each agent is currently heading toward
    std::vector<int> currentWaypointIndex;

    // Resize all vectors to hold n elements.
    void resize(size_t n) {
        x.resize(n);
        y.resize(n);
        destX.resize(n);
        destY.resize(n);
        desiredX.resize(n);
        desiredY.resize(n);
        currentWaypointIndex.resize(n);
    }
};

#endif // AGENTS_SOA_H
