
#ifndef HEATMAP_PAR_H
#define HEATMAP_PAR_H

#include "ped_model.h"
#include <vector>
#include "ped_agent.h"

#ifdef __cplusplus
extern "C" {
#endif
// Allocate device memory once at the start.
void setupCudaHeatmap(int numAgents);

// Run the heatmap update (called every tick).
void updateHeatMapCuda(int *heatmap, int *scaled_heatmap, int *blurred_heatmap, 
                       int *desiredXs, int *desiredYs, int numAgents);

// Free device memory at the end.
void cleanupCudaHeatmap(void);

#ifdef __cplusplus
}
#endif

#endif 