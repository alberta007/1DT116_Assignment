// tick_soa.cpp
#include "soa_tick.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <cstdio>

void tickSoA(AgentsSoA &agents, const WaypointsSoA &waypoints) {
    size_t numAgents = agents.x.size();
    size_t limit = (numAgents / 8) * 8; // multiple of 8

    // We'll assume we have exactly 2 waypoints in 'waypointsSoA' for demonstration
    // So we can do "newWIdx = wIdx ^ 1"
    // If you have more, you'd do a different approach for (wIdx+1)%W

    __m256 eps = _mm256_set1_ps(1e-6f);

    for (size_t i = 0; i < limit; i += 8) {
        // (1) Load each agent's current waypoint index (int)
        __m256i oldWIdx = _mm256_loadu_si256((__m256i*)&agents.currentWaypointIndex[i]);

        // (2) Gather wr, wx, wy from waypointsSoA using oldWIdx
        // Each lane in oldWIdx picks a different index into waypoints.r[], .x[], .y[]
       
        __m256 wrV = _mm256_i32gather_ps(waypoints.r.data(), oldWIdx, 4);
        __m256 wxV = _mm256_i32gather_ps(waypoints.x.data(), oldWIdx, 4);
        __m256 wyV = _mm256_i32gather_ps(waypoints.y.data(), oldWIdx, 4);

        // (3) Load agent positions as floats
        __m256 axV = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i const*)&agents.x[i]));
        __m256 ayV = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i const*)&agents.y[i]));

        // (4) Compute dist^2
        __m256 dxV = _mm256_sub_ps(wxV, axV);
        __m256 dyV = _mm256_sub_ps(wyV, ayV);
        __m256 dist2V = _mm256_fmadd_ps(dxV, dxV, _mm256_mul_ps(dyV, dyV));

        // Compare dist2 < (wr^2)
        __m256 wr2V   = _mm256_mul_ps(wrV, wrV);
        __m256 maskV  = _mm256_cmp_ps(dist2V, wr2V, _CMP_LT_OS); 
        // maskV lanes = all 1 bits if arrived, 0 bits if not

        // (5) For lanes that arrived: wIdx' = wIdx ^ 1 (flip 0->1 or 1->0)
        // For lanes that didn't, keep oldWIdx
        __m256i onesV = _mm256_set1_epi32(1);
        __m256i newWIdx = _mm256_xor_si256(oldWIdx, onesV);  // flip
        __m256i finalWIdx = _mm256_blendv_epi8(
                                oldWIdx,        // if mask=0
                                newWIdx,        // if mask=1
                                _mm256_castps_si256(maskV)
                            );
        
        // Store finalWIdx back
        _mm256_storeu_si256((__m256i*)&agents.currentWaypointIndex[i], finalWIdx);

        // (6) Now gather the NEW waypoint coords for finalWIdx
        __m256 newWxV = _mm256_i32gather_ps(waypoints.x.data(), finalWIdx, 4);
        __m256 newWyV = _mm256_i32gather_ps(waypoints.y.data(), finalWIdx, 4);

        // We'll store them into (destX, destY) so the next movement step uses them
        // But to do a single-step movement now, we can just "move 1 unit" toward them:

        // (7) diffX, diffY from current pos -> new final waypoint
        __m256 diffX = _mm256_sub_ps(newWxV, axV);
        __m256 diffY = _mm256_sub_ps(newWyV, ayV);
        __m256 diffX2= _mm256_mul_ps(diffX, diffX);
        __m256 diffY2= _mm256_mul_ps(diffY, diffY);
        __m256 sum    = _mm256_add_ps(diffX2, diffY2);
        __m256 len    = _mm256_sqrt_ps(sum);
        __m256 lenSafe= _mm256_max_ps(len, eps);

        __m256 normX = _mm256_div_ps(diffX, lenSafe);
        __m256 normY = _mm256_div_ps(diffY, lenSafe);

        __m256 nextXf = _mm256_add_ps(axV, normX); 
        __m256 nextYf = _mm256_add_ps(ayV, normY);

        // Convert to int
        __m256i nextXi = _mm256_cvtps_epi32(nextXf);
        __m256i nextYi = _mm256_cvtps_epi32(nextYf);

        // (8) Store into SoA
        _mm256_storeu_si256((__m256i*)&agents.desiredX[i], nextXi);
        _mm256_storeu_si256((__m256i*)&agents.desiredY[i], nextYi);

        // Also commit them to the agent's current position
        _mm256_storeu_si256((__m256i*)&agents.x[i], nextXi);
        _mm256_storeu_si256((__m256i*)&agents.y[i], nextYi);

        // You can also store the final waypoint coords into (destX, destY)
        // if you want the next iteration's movement to use them:
        // But typically we do that once we do the arrival check next time
        // or if we want the agent to keep traveling multiple steps in a single frame, etc.
        _mm256_storeu_ps(&agents.destX[i], newWxV);
        _mm256_storeu_ps(&agents.destY[i], newWyV);
    }

    // (9) Handle remainder in scalar for indices [limit..numAgents-1]
    for (size_t i = limit; i < numAgents; i++) {
        // same logic scalar for leftover
        int wIdx = agents.currentWaypointIndex[i];
        float wx = waypoints.x[wIdx];
        float wy = waypoints.y[wIdx];
        float wr = waypoints.r[wIdx];

        float ax = (float)agents.x[i];
        float ay = (float)agents.y[i];
        float dx = wx - ax;
        float dy = wy - ay;
        float dist2 = dx*dx + dy*dy;
        if (dist2 < wr*wr) {
            wIdx ^= 1; // flip if 2 total wpts
            agents.currentWaypointIndex[i] = wIdx;
            wx = waypoints.x[wIdx];
            wy = waypoints.y[wIdx];
        }
        // move one step
        float len = std::sqrt(dx*dx + dy*dy);
        if (len < 1e-6f) len=1.0f;
        float nx = dx/len;
        float ny = dy/len;
        int newX = (int)std::round(ax + nx);
        int newY = (int)std::round(ay + ny);
        agents.x[i] = newX;
        agents.y[i] = newY;
        agents.destX[i] = wx;
        agents.destY[i] = wy;
        agents.desiredX[i] = newX;
        agents.desiredY[i] = newY;
    }
}
