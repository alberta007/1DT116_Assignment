#include "ExportSimulationSoa.h"
#include <cstdint> // for int16_t and int32_t

using namespace std;

ExportSimulationSoA::ExportSimulationSoA(Ped::Model &model_, int maxSteps, std::string outputFilename_) 
    : Simulation(model_, maxSteps), outputFilename(outputFilename_)
{
    file = std::ofstream(outputFilename.c_str(), std::ios::binary);
    file.write(reinterpret_cast<const char*>(&maxSimulationSteps), sizeof(maxSimulationSteps));
}

ExportSimulationSoA::~ExportSimulationSoA() {
    file.seekp(0, std::ios::beg);
    file.write(reinterpret_cast<const char*>(&tickCounter), sizeof(tickCounter));
    file.close();
}

void ExportSimulationSoA::serialize()
{
    const AgentsSoA& agentsSoA = model.getAgentsSoA();
    size_t num_agents = agentsSoA.x.size();
    file.write(reinterpret_cast<const char*>(&num_agents), sizeof(num_agents));

    for (size_t i = 0; i < num_agents; ++i) {
        int16_t x = static_cast<int16_t>(agentsSoA.x[i]);
        int16_t y = static_cast<int16_t>(agentsSoA.y[i]);

        file.write(reinterpret_cast<const char *>(&x), sizeof(x));
        file.write(reinterpret_cast<const char *>(&y), sizeof(y));
    }

    int16_t height = HEATMAP_HEIGHT;
    int16_t width = HEATMAP_WIDTH;
    const int* const* heatmap = model.getHeatmap();

    unsigned long heatmap_start = 0xFFFF0000FFFF0000;
    file.write(reinterpret_cast<const char*>(&heatmap_start), sizeof(heatmap_start));

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int ARGBvalue = heatmap[i][j];
            int8_t Avalue = (ARGBvalue >> 24) & ((1 << 8)-1);
            file.write(reinterpret_cast<const char*>(&Avalue), sizeof(Avalue));
        }
    }
    file.flush();
}

void ExportSimulationSoA::runSimulation()
{
    for (int i = 0; i < maxSimulationSteps; i++) {
        tickCounter++;
        model.tick();
        serialize();
    }
}
