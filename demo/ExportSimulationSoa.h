#ifndef EXPORTSIMULATION_SOA_H
#define EXPORTSIMULATION_SOA_H

#include "Simulation.h"
#include <fstream>
#include <string>

#define HEATMAP_WIDTH 160 * 5
#define HEATMAP_HEIGHT 120 * 5
#define HEATMAP_SKIP 5

class ExportSimulationSoA : public Simulation {
public:
    ExportSimulationSoA(Ped::Model &model, int maxSteps, std::string outputFilename);
    ~ExportSimulationSoA();

    void serialize();
    void runSimulation();

private:
    std::string outputFilename;
    std::ofstream file;
    int tickCounter = 0;
};

#endif // EXPORTSIMULATION_SOA_H
