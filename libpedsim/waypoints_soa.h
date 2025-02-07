#ifndef WAYPOINTS_SOA_H
#define WAYPOINTS_SOA_H

#include <vector>

struct WaypointsSoA {
    // Coordinates of each waypoint
    // Example: x[3] = 5, y[3] = 7 means waypoint 3 is at position (5, 7)
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
