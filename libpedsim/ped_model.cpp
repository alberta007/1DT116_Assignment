#include "ped_model.h"
#include "ped_waypoint.h"
#include <iostream>
#include <stack>
#include <algorithm>
#include <omp.h>
#include <thread>
#include <immintrin.h>  // For AVX/SSE SIMD instructions

#ifndef NOCDUA
#include "cuda_testkernel.h"
#endif

#include <stdlib.h>

void Ped::Model::setup(std::vector<Ped::Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario, IMPLEMENTATION implementation) {
#ifndef NOCUDA
    cuda_test();
#else
    std::cout << "Not compiled for CUDA" << std::endl;
#endif

    agents = std::vector<Ped::Tagent*>(agentsInScenario.begin(), agentsInScenario.end());
    destinations = std::vector<Ped::Twaypoint*>(destinationsInScenario.begin(), destinationsInScenario.end());
    this->implementation = implementation;

    // Prepare SoA data structures for SIMD
    agentX.resize(agents.size());
    agentY.resize(agents.size());
    agentDesiredX.resize(agents.size());
    agentDesiredY.resize(agents.size());
    
    for (size_t i = 0; i < agents.size(); i++) {
        agentX[i] = agents[i]->getX();
        agentY[i] = agents[i]->getY();
    }

    setupHeatmapSeq();
}

void Ped::Model::tick() {
    switch (implementation) {
        case 1: { // SIMD vectorization
            size_t totalAgents = agents.size();
            size_t i = 0;

            for (; i + 4 <= totalAgents; i += 4) {
                __m128i xPos = _mm_loadu_si128((__m128i*)&agentX[i]);
                __m128i yPos = _mm_loadu_si128((__m128i*)&agentY[i]);

                // Compute next desired position
                for (int j = 0; j < 4; ++j) {
                    agents[i + j]->computeNextDesiredPosition();
                    agentDesiredX[i + j] = agents[i + j]->getDesiredX();
                    agentDesiredY[i + j] = agents[i + j]->getDesiredY();
                }

                __m128i xDesired = _mm_loadu_si128((__m128i*)&agentDesiredX[i]);
                __m128i yDesired = _mm_loadu_si128((__m128i*)&agentDesiredY[i]);

                _mm_storeu_si128((__m128i*)&agentX[i], xDesired);
                _mm_storeu_si128((__m128i*)&agentY[i], yDesired);
            }

            // Process remaining agents sequentially
            for (; i < totalAgents; ++i) {
                agents[i]->computeNextDesiredPosition();
                agentX[i] = agents[i]->getDesiredX();
                agentY[i] = agents[i]->getDesiredY();
            }
            break;
        }
        
        case 2: { // OpenMP parallelization
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < agents.size(); ++i) {
                agents[i]->computeNextDesiredPosition();
                agentX[i] = agents[i]->getDesiredX();
                agentY[i] = agents[i]->getDesiredY();
            }
            break;
        }
        
        case 3: { // C++ threads parallelization
            const char* envstr = std::getenv("CXX_NUM_THREADS");
            int numThreads = envstr ? std::stoi(envstr) : 1;
            numThreads = std::min(numThreads, static_cast<int>(agents.size()));
            
            std::vector<std::thread> threads;
            threads.reserve(numThreads);
            int blockSize = agents.size() / numThreads;
            int remainder = agents.size() % numThreads;
            
            auto worker = [&](int start, int end) {
                for (int i = start; i < end; ++i) {
                    agents[i]->computeNextDesiredPosition();
                    agentX[i] = agents[i]->getDesiredX();
                    agentY[i] = agents[i]->getDesiredY();
                }
            };
            
            int startIndex = 0;
            for (int t = 0; t < numThreads; ++t) {
                int endIndex = startIndex + blockSize + (t < remainder ? 1 : 0);
                threads.emplace_back(worker, startIndex, endIndex);
                startIndex = endIndex;
            }
            for (auto &thread : threads) {
                thread.join();
            }
            break;
        }
        
        case 4: { // Sequential version
            for (size_t i = 0; i < agents.size(); ++i) {
                agents[i]->computeNextDesiredPosition();
                agentX[i] = agents[i]->getDesiredX();
                agentY[i] = agents[i]->getDesiredY();
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

	// Retrieve their positions
	std::vector<std::pair<int, int> > takenPositions;
	for (std::set<const Ped::Tagent*>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt) {
		std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
		takenPositions.push_back(position);
	}

	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself
	std::vector<std::pair<int, int> > prioritizedAlternatives;
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
	else {
		// Agent wants to walk diagonally
		p1 = std::make_pair(pDesired.first, agent->getY());
		p2 = std::make_pair(agent->getX(), pDesired.second);
	}
	prioritizedAlternatives.push_back(p1);
	prioritizedAlternatives.push_back(p2);

	// Find the first empty alternative position
	for (std::vector<pair<int, int> >::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it) {

		// If the current position is not yet taken by any neighbor
		if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end()) {

			// Set the agent's position 
			agent->setX((*it).first);
			agent->setY((*it).second);

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
set<const Ped::Tagent*> Ped::Model::getNeighbors(int x, int y, int dist) const {

	// create the output list
	// ( It would be better to include only the agents close by, but this programmer is lazy.)	
	return set<const Ped::Tagent*>(agents.begin(), agents.end());
}

void Ped::Model::cleanup() {
	// Nothing to do here right now. 
}

Ped::Model::~Model()
{
	std::for_each(agents.begin(), agents.end(), [](Ped::Tagent *agent){delete agent;});
	std::for_each(destinations.begin(), destinations.end(), [](Ped::Twaypoint *destination){delete destination; });
}
