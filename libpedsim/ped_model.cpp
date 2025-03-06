//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_model.h"
#include "ped_waypoint.h"
#include "ped_model.h"
#include <iostream>
#include <stack>
#include <algorithm>
#include <omp.h>
#include <thread>

#ifndef NOCDUA
#include "cuda_testkernel.h"
#include "tick_cuda.h"
#endif

#include <stdlib.h>
#include "soa_agent.h"
#include <immintrin.h>
#include <cmath>
#include <mutex>

#include "soa_tick.h"

const int WORLDSIZE_X = 160;
const int WORLDSIZE_Y = 120;
std::mutex mtx;

// Constructor: Sets the standard values for the Model when running SEQ, OMP or PTHREAD
void Ped::Model::setup(std::vector<Ped::Tagent *> agentsInScenario, std::vector<Twaypoint *> destinationsInScenario, IMPLEMENTATION implementation)
{
#ifndef NOCUDA
	// Convenience test: does CUDA work on this machine?
	cuda_test();
#else
	std::cout << "Not compiled for CUDA" << std::endl;
#endif
	// Set
	agents = std::vector<Ped::Tagent *>(agentsInScenario.begin(), agentsInScenario.end());

	// Set up destinations
	destinations = std::vector<Ped::Twaypoint *>(destinationsInScenario.begin(), destinationsInScenario.end());

	// Sets the chosen implemenation. Standard in the given code is SEQ

	int regionSizeX = WORLDSIZE_X / 4;
	int regionSizeY = WORLDSIZE_Y / 2;
	int startX = 0;
	int startY = 0;

	
	this->addRegion(new Region(startX, startX + regionSizeX, startY, startY + regionSizeY, 1));
    this->addRegion(new Region(startX + regionSizeX, startX + 2*regionSizeX, startY, startY + regionSizeY, 2));
    this->addRegion(new Region(startX + 2*regionSizeX, startX + 3*regionSizeX, startY, startY + regionSizeY, 3));
    this->addRegion(new Region(startX + 3*regionSizeX, WORLDSIZE_X, startY, startY + regionSizeY, 4));

    // Bottom row (regions 5-8)
    this->addRegion(new Region(startX, startX + regionSizeX, startY + regionSizeY, WORLDSIZE_Y, 5));
    this->addRegion(new Region(startX + regionSizeX, startX + 2*regionSizeX, startY + regionSizeY, WORLDSIZE_Y, 6));
    this->addRegion(new Region(startX + 2*regionSizeX, startX + 3*regionSizeX, startY + regionSizeY, WORLDSIZE_Y, 7));
    this->addRegion(new Region(startX + 3*regionSizeX, WORLDSIZE_X, startY + regionSizeY, WORLDSIZE_Y, 8));

	for (auto agent : agents)
	{
		this->placeAgentInRegion(agent);
	}

	this->listRegions();

	this->implementation = implementation;

	// Set up heatmap (relevant for Assignment 4)
	setupHeatmapSeq();
}

// Setup for SIMD implementation
void Ped::Model::setup(const AgentsSoA &agentsSoA,
					   const WaypointsSoA &waypointsSoA,
					   IMPLEMENTATION implementation)
{
#ifndef NOCUDA
	cuda_test();
#else
	std::cout << "Not compiled for CUDA" << std::endl;
#endif
	// Store the SoA representation directly.
	this->agentsSoA = agentsSoA;
	this->waypointsSoA = waypointsSoA;

	// Store the chosen implementation.
	this->implementation = implementation;

	// Set up heatmap.
	setupHeatmapSeq();
}

void Ped::Model::tick()
{

	switch (implementation)
	{

	// case CUDA:
	// {
	// 	// Call the CUDA implementation of tick (in tick_cuda.cpp)
	// 	tickCuda(agentsSoA, waypointsSoA);
	// 	break;
	// }

	case VECTOR:
	{
		// Call the SoA implementation of tick (in soa_tick.cpp)
		tickSoA(agentsSoA, waypointsSoA);

		break;
	}

	case OMP:
	{
		// Parallelization using OpenMP where each region is processed in parallel
		// default(shared) for regions, but each thread has private agent pointers

		#pragma omp parallel for schedule(dynamic)
		for (size_t i = 0; i < this->regions.size(); i++)
		{

			auto region = this->regions[i];
			std::vector<Tagent *> agentsToProcess = region->agentsInRegion;

			for (auto agent : agentsToProcess)
			{
				
					agent->computeNextDesiredPosition();

				agent->computeNextDesiredPosition();
				{

					move(agent);
				}
			}
			
		}

		updateHeatmapSeq();
		break;
	}

	case PTHREAD:
	{
		// C++ threads parallelization
		// printf("Using C++ threads\n");

		// 1. Determine how many threads to use
		const char *envstr = std::getenv("CXX_NUM_THREADS");
		int numThreads = 1; // default
		if (envstr != nullptr)
		{
			numThreads = std::stoi(envstr);
		}
		// printf("Using %d threads\n", numThreads);
		// 2. Calculate the total number of agents
		int totalAgents = static_cast<int>(agents.size());

		// 3. Limit the number of threads so we don't spawn more threads than agents
		numThreads = std::min(numThreads, totalAgents);

		// 4. Divide work among the threads
		int blockSize = totalAgents / numThreads;
		int remainder = totalAgents % numThreads;

		// We'll store our thread objects here
		std::vector<std::thread> threads;
		threads.reserve(numThreads);

		// 5. Worker function to process a subset of agents
		// Using [=] to capture by value is often safer, but [&, this] could also work
		auto worker = [=](int start, int end)
		{
			for (int i = start; i < end; i++)
			{
				agents[i]->computeNextDesiredPosition();
				agents[i]->setX(agents[i]->getDesiredX());
				agents[i]->setY(agents[i]->getDesiredY());
			}
		};

		// 6. Create the threads
		int startIndex = 0;
		for (int t = 0; t < numThreads; t++)
		{
			int endIndex = startIndex + blockSize + (t < remainder ? 1 : 0);
			threads.emplace_back(worker, startIndex, endIndex);
			startIndex = endIndex;
		}

		// 7. Join the threads to ensure all work completes
		for (auto &thread : threads)
		{
			thread.join();
		}

		break;
	}

	case SEQ:
	{
		for (auto region : this->regions)
		{
			for (auto agent : region->agentsInRegion)
			{
				// Get the next desired position
				agent->computeNextDesiredPosition();
				// Update the position
				// agent->setX(agent->getDesiredX());
				// agent->setY(agent->getDesiredY());
				move(agent);
			}
		}

		updateHeatmapSeq();
		break;
	}
	}
}
////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

// Moves the agent to the next desired position. If the desired cell is taken,
// it tries alternative nearby cells.
void Ped::Model::move(Tagent *agent)
{
    // Retrieve the current region for the agent.
    Region* currRegion = this->getAgentCurrentRegion(agent);
    
    // Determine if the agent is near the region border.
    // "Near" is defined here as being within 2 cells of any boundary.
    bool nearBorder = false;
    int x = agent->getX();
    int y = agent->getY();
    if (x <= currRegion->startX + 2 || x >= currRegion->endX - 2 ||
        y <= currRegion->startY + 2 || y >= currRegion->endY - 2)
    {
        nearBorder = true;
    }
    
	Region* targetRegion = determineTargetRegion(agent);

	if (targetRegion != currRegion) {
		// Cross-region move: lock both regions.
		if (currRegion->regionID < targetRegion->regionID) {
			std::lock_guard<std::mutex> lockCurr(*(currRegion->lock));
			std::lock_guard<std::mutex> lockTarget(*(targetRegion->lock));
			performMove(agent, currRegion);
		} else {
			std::lock_guard<std::mutex> lockTarget(*(targetRegion->lock));
			std::lock_guard<std::mutex> lockCurr(*(currRegion->lock));
			performMove(agent, currRegion);
		}
	} else {
		// Move remains in the same region.
		// Optionally, you might also check if the move is near the border
		// and then lock even if it doesn't change regions.
		if (nearBorder) {
			std::lock_guard<std::mutex> guard(*(currRegion->lock));
			performMove(agent, currRegion);
		} else {
			performMove(agent, currRegion);
		}
	}

}

// Helper function that implements the actual move logic.
void Ped::Model::performMove(Tagent *agent, Region *currRegion)
{
    // Gather the positions of nearby agents.
    std::set<const Tagent*> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);
    std::vector<std::pair<int, int>> takenPositions;
    for (auto neighbor : neighbors)
    {
        takenPositions.push_back(std::make_pair(neighbor->getX(), neighbor->getY()));
    }
    
    // Compute candidate positions (desired position plus two alternatives).
    std::vector<std::pair<int, int>> prioritizedAlternatives;
    std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
    prioritizedAlternatives.push_back(pDesired);
    
    int diffX = pDesired.first - agent->getX();
    int diffY = pDesired.second - agent->getY();
    std::pair<int, int> p1, p2;
    if (diffX == 0 || diffY == 0)
    {
        // For straight movements, adjust using diffX/diffY.
        p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
        p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
    }
    else
    {
        // For diagonal movement, try altering one coordinate at a time.
        p1 = std::make_pair(pDesired.first, agent->getY());
        p2 = std::make_pair(agent->getX(), pDesired.second);
    }
    prioritizedAlternatives.push_back(p1);
    prioritizedAlternatives.push_back(p2);
    
    // Try each candidate alternative until one is free.
    for (auto alt : prioritizedAlternatives)
    {
        if (std::find(takenPositions.begin(), takenPositions.end(), alt) == takenPositions.end())
        {
            // If the candidate cell is inside the current region, update the agent's position.
            if (currRegion->contains(alt.first, alt.second))
            {
                agent->setX(alt.first);
                agent->setY(alt.second);
            }
            else
            {
                // If the move is out of the current region,
                // remove the agent from the current region, update its position, and add it to the new region.
                removeAgentFromRegion(agent);
                agent->setX(alt.first);
                agent->setY(alt.second);
                placeAgentInRegion(agent);
            }
            break;
        }
    }
}



/// Returns the list of neighbors within dist of the point x/y. This
/// can be the position of an agent, but it is not limited to this.
/// \date    2012-01-29
/// \return  The list of neighbors
/// \param   x the x coordinate
/// \param   y the y coordinate
/// \param   dist the distance around x/y that will be searched for agents (search field is a square in the current implementation)
set<const Ped::Tagent *> Ped::Model::getNeighbors(int x, int y, int dist) const
{

	// create the output list
	// ( It would be better to include only the agents close by, but this programmer is lazy.)
	return set<const Ped::Tagent *>(agents.begin(), agents.end());
}

void Ped::Model::cleanup()
{
	// Nothing to do here right now.
}

Ped::Model::~Model()
{
	// Clean up heatmap memory
	freeHeatmapMemory();
}
