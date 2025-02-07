// tick_soa.cpp
#include "soa_tick.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <cstdio>


// Function to update the waypoint index for lanes that have arrived.

static inline __m256i updateWaypointIndices(__m256i oldWIdx, int numWaypoints) {
    // Create a vector of ones.
    __m256i one = _mm256_set1_epi32(1); // [1, 1, 1, 1, 1, 1, 1, 1]
    // Increment the old waypoint index.
    __m256i inc = _mm256_add_epi32(oldWIdx, one); // [1, 2, 1, 3, 2, 1, 1, 2]
    // Create a mask: true (all ones) if inc > (numWaypoints - 1)
    __m256i threshold = _mm256_set1_epi32(numWaypoints - 1); // [3, 3, 3, 3, 3, 3, 3, 3] if numWaypoints = 4
    __m256i mask = _mm256_cmpgt_epi32(inc, threshold); // [0, 0, 0, 1, 0, 0, 0, 0]
    // Where mask is true, we want 0; otherwise, use inc.
    __m256i newWIdx = _mm256_blendv_epi8(inc, _mm256_setzero_si256(), mask); // [1, 2, 1, 0, 2, 1, 1, 2]
    return newWIdx;
}

void tickSoA(AgentsSoA &agents, const WaypointsSoA &waypoints) {
    // Number of waypoints
    const int numWaypoints = (int) waypoints.x.size();
    // Number of agents
    size_t numAgents = agents.x.size();
    // Limit to process in blocks of 8 agents  
    size_t limit = (numAgents / 8) * 8;
    // Small constant to avoid division by zero
    __m256 eps = _mm256_set1_ps(1e-6f);

    for (size_t i = 0; i < limit; i += 8) {
        // (1) Load each agent's current waypoint index (int) for 8 agents.
        __m256i oldWIdx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&agents.currentWaypointIndex[i]));

        // (2) Gather waypoint data (radius, x, y) using the current indices.
        // Where .data() returns a pointer to the first element of the vector.
        // The 4 means that each element is 4 bytes (float).
        // Example: if oldWIdx = [0, 1, 0, 2, 1, 0, 0, 1], and waypoints.r = [10, 5, 3],
        // then wrV = [10, 5, 10, 3, 5, 10, 10, 5].
        __m256 wrV = _mm256_i32gather_ps(waypoints.r.data(), oldWIdx, 4);
        __m256 wxV = _mm256_i32gather_ps(waypoints.x.data(), oldWIdx, 4);
        __m256 wyV = _mm256_i32gather_ps(waypoints.y.data(), oldWIdx, 4);

        // (3) Load agent positions as floats.
        //    Since agents.x and agents.y are int arrays, we need to convert them to floats.
        // They are ints because they represent positions in a grid. Needed to plot agents in the GUI.
        __m256 axV = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(&agents.x[i])));
        __m256 ayV = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(&agents.y[i])));

        // (4) Compute distance squared from agent positions to the waypoint.
        // Basically, we compute the Euclidean distance between the agent and the waypoint. The hypotenuse of a triangle.
        __m256 dxV = _mm256_sub_ps(wxV, axV);
        __m256 dyV = _mm256_sub_ps(wyV, ayV);
        // The _mm256_fmadd_ps(a,b,c) function computes (a * b) + c for each lane.
        __m256 dist2V = _mm256_fmadd_ps(dxV, dxV, _mm256_mul_ps(dyV, dyV));

        // (5) Compute waypoint radius squared.
        __m256 wr2V = _mm256_mul_ps(wrV, wrV);

        // (6) Check if any lane satisfies the condition. Means that some agent has arrived at the waypoint. Then update the waypoint index.
        __m256 maskV = _mm256_cmp_ps(dist2V, wr2V, _CMP_LT_OS); // check if dist2 < wr2
        int maskBits = _mm256_movemask_ps(maskV);  // each bit corresponds to a lane
        
        // New waypoint coordinates.
        __m256 newWxV, newWyV;
        // Check if any lane is true which means that some agent has arrived at the waypoint.
        if (maskBits != 0) {
            // Compute new waypoint indices: newWIdx = (oldWIdx + 1) mod numWaypoints.
            __m256i newWIdx = updateWaypointIndices(oldWIdx, numWaypoints);
            // Blend: if mask is true (agent arrived), use newWIdx; else, keep oldWIdx.
            __m256i finalWIdx = _mm256_blendv_epi8(oldWIdx, newWIdx, _mm256_castps_si256(maskV));
            // Store the updated waypoint indices back.
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&agents.currentWaypointIndex[i]), finalWIdx);
            // Gather the new waypoint coordinates using the updated indices.
            newWxV = _mm256_i32gather_ps(waypoints.x.data(), finalWIdx, 4);
            newWyV = _mm256_i32gather_ps(waypoints.y.data(), finalWIdx, 4);
        }
        else {
            // No lane has "arrived"; use the current waypoint coordinates.
            newWxV = wxV;
            newWyV = wyV;
        }

        // (7) Compute the difference vector from the current agent positions to the chosen waypoint.
        __m256 diffX = _mm256_sub_ps(newWxV, axV);
        __m256 diffY = _mm256_sub_ps(newWyV, ayV);
        __m256 diffX2 = _mm256_mul_ps(diffX, diffX);
        __m256 diffY2 = _mm256_mul_ps(diffY, diffY);
        __m256 sum = _mm256_add_ps(diffX2, diffY2);
        __m256 len = _mm256_sqrt_ps(sum);
        __m256 lenSafe = _mm256_max_ps(len, eps);

        // (8) Normalize the difference vector to obtain a unit step.
        __m256 normX = _mm256_div_ps(diffX, lenSafe);
        __m256 normY = _mm256_div_ps(diffY, lenSafe);
        __m256 nextXf = _mm256_add_ps(axV, normX);
        __m256 nextYf = _mm256_add_ps(ayV, normY);

        // (9) Convert the new positions from float back to integers (with rounding).
        __m256i nextXi = _mm256_cvtps_epi32(nextXf);
        __m256i nextYi = _mm256_cvtps_epi32(nextYf);

        // (10) Store the new positions into the desired and current position arrays.
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&agents.x[i]), nextXi);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&agents.y[i]), nextYi);
      
    }

    // Process any remaining agents in a scalar loop.
    for (size_t i = limit; i < numAgents; i++) {
        int wIdx = agents.currentWaypointIndex[i];
        float wx = waypoints.x[wIdx];
        float wy = waypoints.y[wIdx];
        float wr = waypoints.r[wIdx];

        float ax = static_cast<float>(agents.x[i]);
        float ay = static_cast<float>(agents.y[i]);
        float dx = wx - ax;
        float dy = wy - ay;
        float dist2 = dx*dx + dy*dy;
        if (dist2 < wr*wr) {
            wIdx = (wIdx + 1) % numWaypoints;
            agents.currentWaypointIndex[i] = wIdx;
            wx = waypoints.x[wIdx];
            wy = waypoints.y[wIdx];
        }
        float len = std::sqrt(dx*dx + dy*dy);
        if (len < 1e-6f) len = 1.0f;
        float nx = dx/len;
        float ny = dy/len;
        int newX = static_cast<int>(std::round(ax + nx));
        int newY = static_cast<int>(std::round(ay + ny));
        agents.x[i] = newX;
        agents.y[i] = newY;
    }
}
