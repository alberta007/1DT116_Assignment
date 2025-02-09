#ifndef WAYPOINTS_SOA_H
#define WAYPOINTS_SOA_H

#include <vector>

/**
 * Structure of Arrays (SoA) representation of waypoints.
 * We store the x, y, and r coordinates of each waypoint in separate arrays.
 * This is mainly used for storing the waypoints so that they can be easily accessed when
 * computing the next desired position of an agent.
 */

struct WaypointsSoA {
    // Coordinates of each waypoint
    // Example: x[2] = 5, y[2] = 3 means waypoint 3 is at position (5, 3)
    // [5, 3, 5, 7, 3, 2, 1, 0, 0, 1] means 10 waypoints are at x positions 5, 3, 5, ...
    // [7, 2, 3, 7, 2, 1, 0, 1, 2, 2] means 10 waypoints are at y positions 7, 2, 3, ...
    std::vector<float> x;
    std::vector<float> y;
    
    // Optional: radius if you want each waypoint to have a threshold
    // [10,10,5,3,8,8,8,8,8,8] means 10 waypoints have radii 10, 10, 5, 3, ...
    std::vector<float> r;

    void resize(size_t n) {
        x.resize(n);
        y.resize(n);
        r.resize(n);
    }
};

#endif
