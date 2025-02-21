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

	int regionSizeX = WORLDSIZE_X / 2;
	int regionSizeY = WORLDSIZE_Y / 2;
	int startX = 0;
	int startY = 0;

	this->addRegion(Region(startX, regionSizeX, startY, regionSizeY,1));
	this->addRegion(Region(regionSizeX, WORLDSIZE_X, startY, regionSizeY,2));

	this->addRegion(Region(startX, regionSizeX, regionSizeY, WORLDSIZE_Y,3));
	this->addRegion(Region(regionSizeX, WORLDSIZE_X, regionSizeY, WORLDSIZE_Y,4));

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

	case CUDA:
	{
		// Call the CUDA implementation of tick (in tick_cuda.cpp)
		tickCuda(agentsSoA, waypointsSoA);
		break;
	}

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
		#pragma omp parallel for schedule(dynamic) default(shared)
		for (size_t i = 0; i < this->regions.size(); i++) {
			auto region = this->regions[i];
			// Temporary vector
			std::vector<Tagent*> agentsToProcess;			
			std::cout << "\nProcessing region: " << region->regionid() << std::endl;
			agentsToProcess = region->agentsInRegion; // Create copy
	
			// Process each agent in this region
			for (auto agent : agentsToProcess) {
				agent->computeNextDesiredPosition();

				move(agent);
			}
		}
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

		break;
	}
	}
}
////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
void Ped::Model::move(Ped::Tagent *agent)
{
	// Search for neighboring agents
	set<const Ped::Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);

	Region *currRegion = this->getAgentCurrentRegion(agent);
	// Retrieve their positions
	std::vector<std::pair<int, int>> takenPositions;
	for (std::set<const Ped::Tagent *>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt)
	{
		std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
		takenPositions.push_back(position);
	}

	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself
	std::vector<std::pair<int, int>> prioritizedAlternatives;
	std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
	prioritizedAlternatives.push_back(pDesired);

	int diffX = pDesired.first - agent->getX();
	int diffY = pDesired.second - agent->getY();
	std::pair<int, int> p1, p2;
	if (diffX == 0 || diffY == 0)
	{
		// Agent wants to walk straight to North, South, West or East
		p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
		p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
	}
	else
	{
		// Agent wants to walk diagonally
		p1 = std::make_pair(pDesired.first, agent->getY());
		p2 = std::make_pair(agent->getX(), pDesired.second);
	}
	prioritizedAlternatives.push_back(p1);
	prioritizedAlternatives.push_back(p2);

	switch (implementation)
	{
	case OMP:
	{
		// std::cout<<"RUNNING OMP\n";
		//  Find the first empty alternative position
		for (std::vector<pair<int, int>>::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it)
		{

			// If the current position is not yet taken by any neighbor
			if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end())
			{

				if (currRegion->contains((*it).first, (*it).second))
				{
					// Set the agent's position
					agent->setX((*it).first);
					agent->setY((*it).second);
				}			
				else
				{
				{
					mtx.lock();
					this->removeAgentFromRegion(agent);
					agent->setX((*it).first);
					agent->setY((*it).second);
					this->placeAgentInRegion(agent);
					mtx.unlock();
				}
				}

				break;
			}
		}

		break;
	}
	default:
	{
		// std::cout<<"RUNNING SEQ\n";
		//  Find the first empty alternative position
		for (std::vector<pair<int, int>>::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it)
		{

			// If the current position is not yet taken by any neighbor
			if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end())
			{
				// Set the agent's position
				agent->setX((*it).first);
				agent->setY((*it).second);

				break;
			}
		}
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
