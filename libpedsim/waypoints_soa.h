#ifndef WAYPOINTS_SOA_H
#define WAYPOINTS_SOA_H

#include <vector>

struct WaypointsSoA {
    // Coordinates of each waypoint
    std::vector<float> x;
    std::vector<float> y;
    
    // Optional: radius if you want each waypoint to have a threshold
    std::vector<float> r;

    void resize(size_t n) {
        x.resize(n);
        y.resize(n);
        r.resize(n);
    }
};

#endif
