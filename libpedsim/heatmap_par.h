
#ifndef HEATMAP_PAR_H
#define HEATMAP_PAR_H

#include "ped_model.h"
#include <vector>
#include "ped_agent.h"

// Host function that launches the CUDA kernel.
void updateHeatMapCuda(int *heatmap, int *scaled_heatmap, int *blurred_heatmap, int *desiredXs, int *desiredYs, int numAgents);

#endif 