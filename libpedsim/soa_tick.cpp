// tick_soa.cpp
#include "soa_tick.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <cstdio>

void tickSoA(AgentsSoA &agents, const WaypointsSoA &waypoints) {
    size_t numAgents = agents.x.size();
    //////////////////////////////////////////////////////////////////////////
    // (A) ARRIVAL CHECK (scalar)
    //     For each agent, see if we are within waypointsSoA.r[] of our current
    //     waypoint. If so, pick the next waypoint and update (destX, destY).
    //////////////////////////////////////////////////////////////////////////
    for (size_t i = 0; i < numAgents; i++) {
        int wIdx = agents.currentWaypointIndex[i];
        // If you have multiple waypoints:
        //   wIdx might be out of range if you haven't wrapped it, so be sure
        //   it's valid, or do (wIdx % waypoints.x.size()) if you're looping.
        float wx = waypoints.x[wIdx];
        float wy = waypoints.y[wIdx];
        float wr = waypoints.r[wIdx];

        // Current agent position
        float ax = static_cast<float>(agents.x[i]);
        float ay = static_cast<float>(agents.y[i]);

        // Distance^2 to the waypoint
        float dx = wx - ax;
        float dy = wy - ay;
        float dist2 = dx*dx + dy*dy;

        // If agent is within wr of that waypoint, pick the next
        if (dist2 < (wr * wr)) {
            // For example, cycle to the next waypoint if you have multiple
            // wIdx = (wIdx + 1) % waypoints.x.size();
            // or if it's just 2, do wIdx = 1-wIdx;
            // or if you have a route array, you'd do something else.

            wIdx = (wIdx + 1) % waypoints.x.size();
            agents.currentWaypointIndex[i] = wIdx;
         
            // Now set new (destX, destY) to the new waypoint
            agents.destX[i] = waypoints.x[wIdx];
            agents.destY[i] = waypoints.y[wIdx];
        }
        // else: keep the same waypoint
    }
    size_t limit = (numAgents / 8) * 8;  // Process in blocks of 8 agents

    // SIMD loop: Process blocks of 8 agents using AVX2.
    for (size_t i = 0; i < limit; i += 8) {
        // Load current positions (stored as int) and convert them to float.
        __m256 curX = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i const*)&agents.x[i]));
        __m256 curY = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i const*)&agents.y[i]));
        __m256 destX = _mm256_loadu_ps(&agents.destX[i]);
        __m256 destY = _mm256_loadu_ps(&agents.destY[i]);

        // Compute difference vectors.
        __m256 diffX = _mm256_sub_ps(destX, curX);
        __m256 diffY = _mm256_sub_ps(destY, curY);

        // Compute squared differences.
        __m256 diffX2 = _mm256_mul_ps(diffX, diffX);
        __m256 diffY2 = _mm256_mul_ps(diffY, diffY);
        __m256 sum = _mm256_add_ps(diffX2, diffY2);
        __m256 len = _mm256_sqrt_ps(sum);

        // Avoid division by zero by using an epsilon.
        __m256 eps = _mm256_set1_ps(1e-6f);
        __m256 lenSafe = _mm256_max_ps(len, eps);

        // Compute normalized difference.
        __m256 normX = _mm256_div_ps(diffX, lenSafe);
        __m256 normY = _mm256_div_ps(diffY, lenSafe);

        // Compute new (desired) positions: current position plus normalized step.
        __m256 simdDesiredX = _mm256_add_ps(curX, normX);
        __m256 simdDesiredY = _mm256_add_ps(curY, normY);

        // Convert back to integers with rounding.
        __m256i desiredX_i = _mm256_cvtps_epi32(simdDesiredX);
        __m256i desiredY_i = _mm256_cvtps_epi32(simdDesiredY);

        _mm256_storeu_si256((__m256i*)&agents.desiredX[i], desiredX_i);
        _mm256_storeu_si256((__m256i*)&agents.desiredY[i], desiredY_i);
    }

    // Process any remaining agents in a scalar loop.
    for (size_t i = limit; i < numAgents; ++i) {
        float cx = static_cast<float>(agents.x[i]);
        float cy = static_cast<float>(agents.y[i]);
        float dx = agents.destX[i];
        float dy = agents.destY[i];
        float diffX = dx - cx;
        float diffY = dy - cy;
        float len = std::sqrt(diffX * diffX + diffY * diffY);
        if (len < 1e-6f) { len = 1.0f; }
        float normX = diffX / len;
        float normY = diffY / len;
        int desiredX = static_cast<int>(std::round(cx + normX));
        int desiredY = static_cast<int>(std::round(cy + normY));
        agents.desiredX[i] = desiredX;
        agents.desiredY[i] = desiredY;
    }

    // Update current positions with the computed desired positions.
    // This simulates the agents moving.
    for (size_t i = 0; i < numAgents; i++) {
        agents.x[i] = agents.desiredX[i];
        agents.y[i] = agents.desiredY[i];
    }
}
