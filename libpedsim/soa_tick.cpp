#include "soa_tick.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <cstdio>

// Helper: update a vector of waypoint indices cyclically.
// For each lane, computes new index = (oldIndex + 1) mod numWaypoints.
static inline __m256i updateWaypointIndices(__m256i oldWIdx, int numWaypoints) {
    __m256i one = _mm256_set1_epi32(1);             // [1,1,1,1,1,1,1,1]
    __m256i inc = _mm256_add_epi32(oldWIdx, one);     // Increment each index.
    __m256i threshold = _mm256_set1_epi32(numWaypoints - 1);
    __m256i mask = _mm256_cmpgt_epi32(inc, threshold); // true if (oldIndex+1) > (numWaypoints-1)
    // Where mask is true, wrap around to 0; otherwise use inc.
    __m256i newWIdx = _mm256_blendv_epi8(inc, _mm256_setzero_si256(), mask);
    return newWIdx;
}

void tickSoA(AgentsSoA &agents, const WaypointsSoA &waypoints) {
    // Number of waypoints
    const int numWaypoints = static_cast<int>(waypoints.x.size());
    // Number of agents
    size_t numAgents = agents.x.size();
    // Set a limit to ensure we don't read out of bounds.
    size_t limit = (numAgents / 8) * 8;
    // Small constant to avoid division by zero.
    __m256 eps = _mm256_set1_ps(1e-6f);

    for (size_t i = 0; i < limit; i += 8) {
        // (1) Load current waypoint indices for 8 agents.
        __m256i oldWIdx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&agents.currentWaypointIndex[i]));

        // (2) Load destination data (destR, destX, destY) from agents’ SoA. 
        __m256 wrV = _mm256_loadu_ps(&agents.destR[i]); // Load 8 destination radii of the waypoint the agent is heading towards.
        __m256 wxV = _mm256_loadu_ps(&agents.destX[i]); // Load 8 destination X coordinates. The agent is heading towards this X.
        __m256 wyV = _mm256_loadu_ps(&agents.destY[i]); // Load 8 destination Y coordinates. The agent is heading towards this Y.

        // (3) Load agent positions as floats (convert from int).
        //  We save the x,y as integers in the SoA, but we need them as floats for the calculations.
        // The reason we store them as integers is becuase we want to plot them as integers in the GUI.
        __m256 axV = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(&agents.x[i])));
        __m256 ayV = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(&agents.y[i])));

        // (4) Compute squared distance from current position to destination.
        __m256 dxV = _mm256_sub_ps(wxV, axV); // Compute difference in X. dx = wx - ax
        __m256 dyV = _mm256_sub_ps(wyV, ayV); // Compute difference in Y. dy = wy - ay
        __m256 dist2V = _mm256_fmadd_ps(dxV, dxV, _mm256_mul_ps(dyV, dyV)); // computes a * b + c for each lane.

        // (5) Compute destination radius squared.
        __m256 wr2V = _mm256_mul_ps(wrV, wrV); 

        // (6) Compute a mask: for each lane, true if dist2 < wr^2.
        // This mask will be used to determine which agents have arrived at their destination.
        // cmp_lt_os: less-than comparison, ordered, signaling.
        __m256 maskV = _mm256_cmp_ps(dist2V, wr2V, _CMP_LT_OS); 


        // Prepare temporary arrays for updated destination coordinates for this block.
        // used because we need to update the destination coordinates for agents that have arrived.
        int indices[8];
        float newWx[8], newWy[8], newWr[8];

        int maskBits = _mm256_movemask_ps(maskV);  // Each bit corresponds to a lane.

        // Check if any lane in this block has arrived. If so, update the destination coordinates.
        if (maskBits != 0) {
            // Compute the updated waypoint indices for the entire block.
            __m256i updatedWIdx = updateWaypointIndices(oldWIdx, numWaypoints);
            // Store updated indices into our temporary array.
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(indices), updatedWIdx);
            
            // Pre-store current destination values from vector registers (to avoid repeated stores).
            float curDestX[8], curDestY[8], curDestR[8];
            _mm256_storeu_ps(curDestX, wxV);
            _mm256_storeu_ps(curDestY, wyV);
            _mm256_storeu_ps(curDestR, wrV);
            
            // For each lane in this block:
            for (int j = 0; j < 8; j++) {
                if (maskBits & (1 << j)) {  // If lane j has arrived.
                    int newIdx = indices[j];  // Retrieve the updated index from the temporary array.
                    agents.currentWaypointIndex[i + j] = newIdx; // Write it back into the agents SoA.
                    newWx[j] = waypoints.x[newIdx];  // Get new destination X.
                    newWy[j] = waypoints.y[newIdx];  // Get new destination Y.
                    newWr[j] = waypoints.r[newIdx];  // Get new destination R.
                } else {
                    // Lane j has not arrived; use the current destination.
                    newWx[j] = curDestX[j];
                    newWy[j] = curDestY[j];
                    newWr[j] = curDestR[j];
                }
            }
        } else {
            // No lane in this block has arrived; use the current destination.
            _mm256_storeu_ps(newWx, wxV);
            _mm256_storeu_ps(newWy, wyV);
            _mm256_storeu_ps(newWr, wrV);
            
        }

        // (7) Load the new destination coordinates into vector registers.
        __m256 newWxV = _mm256_loadu_ps(newWx);
        __m256 newWyV = _mm256_loadu_ps(newWy);
        __m256 newWrV = _mm256_loadu_ps(newWr);

        // save the new destination coordinates back to the agents’ SoA.
        _mm256_storeu_ps(&agents.destX[i], newWxV);
        _mm256_storeu_ps(&agents.destY[i], newWyV);
        _mm256_storeu_ps(&agents.destR[i], newWrV);


        
        // (8) Compute the movement step: vector from current position to destination.
        // we calculate the vectors that point from the current position to the destination.
        __m256 diffX = _mm256_sub_ps(newWxV, axV);
        __m256 diffY = _mm256_sub_ps(newWyV, ayV);
        __m256 diffX2 = _mm256_mul_ps(diffX, diffX);
        __m256 diffY2 = _mm256_mul_ps(diffY, diffY);
        __m256 sum = _mm256_add_ps(diffX2, diffY2);
        __m256 len = _mm256_sqrt_ps(sum);
        __m256 lenSafe = _mm256_max_ps(len, eps);
        
        // (9) Normalize the difference vector to obtain a unit step.
        __m256 normX = _mm256_div_ps(diffX, lenSafe);
        __m256 normY = _mm256_div_ps(diffY, lenSafe);
        __m256 nextXf = _mm256_add_ps(axV, normX);
        __m256 nextYf = _mm256_add_ps(ayV, normY);
        
        // (10) Convert the new positions from float back to int (with rounding).
        __m256i nextXi = _mm256_cvtps_epi32(nextXf);
        __m256i nextYi = _mm256_cvtps_epi32(nextYf);
        
        // (11) Store the new positions into the agent arrays.
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
            wr = waypoints.r[wIdx];
        }
        float len = std::sqrt(dx*dx + dy*dy);
        if (len < 1e-6f) len = 1.0f;
        float nx = dx/len;
        float ny = dy/len;
        int newX = static_cast<int>(std::round(ax + nx));
        int newY = static_cast<int>(std::round(ay + ny));
        agents.x[i] = newX;
        agents.y[i] = newY;
        agents.destX[i] = wx;
        agents.destY[i] = wy;
        agents.destR[i] = wr;
    }
}
