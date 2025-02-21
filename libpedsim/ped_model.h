//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// Model coordinates a time step in a scenario: for each
// time step all agents need to be moved by one position if
// possible.
//
#ifndef _ped_model_h_
#define _ped_model_h_

#include <map>
#include <set>
#include <iostream>
#include <vector>
#include <algorithm>

#include "ped_agent.h"
#include "soa_agent.h"
#include "waypoints_soa.h"

namespace Ped{
	class Tagent;

	// The implementation modes for Assignment 1 + 2:
	// chooses which implementation to use for tick()
	enum IMPLEMENTATION { CUDA, VECTOR, OMP, PTHREAD, SEQ };

	class Region {
		public: 
			int startX, endX, startY, endY;

			std::vector<Tagent*> agentsInRegion;

			Region(int x1, int x2, int y1, int y2)
				: startX(x1), endX(x2), startY(y1), endY(y2) {}

			bool contains(int x, int y) const {
				return (x >= startX && x < endX && y >= startY && y < endY);
			}

			void addToAgentsInRegion(Tagent* agent) {
				agentsInRegion.push_back(agent);
			}

			void removeFromAgentsInRegion(Tagent* agent) {
				agentsInRegion.erase(std::remove_if(agentsInRegion.begin(), agentsInRegion.end(),
													[agent](Tagent* agentToRemove) {return agentToRemove->getX() == agent->getX() && agentToRemove->getY() == agent->getY();}),
												agentsInRegion.end());
			}
	};

	class Model
	{
	public:

		// Sets everything up
		void setup(std::vector<Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario,IMPLEMENTATION implementation);

		void setup(const AgentsSoA& agentsSoA,const WaypointsSoA &waypointsSoA, IMPLEMENTATION implementation);
		
		// Coordinates a time step in the scenario: move all agents by one step (if applicable).
		void tick();

		// Returns the agents of this scenario
		const std::vector<Tagent*>& getAgents() const { return agents; };

		// Returns the agents of this scenario in SoA format
		const AgentsSoA& getAgentsSoA() const { return agentsSoA; };

		// Adds an agent to the tree structure
		void placeAgent(const Ped::Tagent *a);

		// Cleans up the tree and restructures it. Worth calling every now and then.
		void cleanup();
		~Model();

		// Returns the heatmap visualizing the density of agents
		int const * const * getHeatmap() const { return blurred_heatmap; };
		int getHeatmapSize() const;

		void addRegion(const Region& region) {
			regions.push_back(new Region(region));
		}

		void listRegions() const {
			for (size_t i = 0; i < regions.size(); ++i) {
				std::cout << "Region " << i << ": (" 
						  << regions[i]->startX << ", " << regions[i]->startY << ") to ("
						  << regions[i]->endX << ", " << regions[i]->endY << "), SIZE OF AGENT: " << regions[i]->agentsInRegion.size() << "\n";
			}
		}

		void placeAgentInRegion(Tagent* agent) {
			for(auto region : regions) {
				if (region->contains(agent->getX(), agent->getY())) {
					region->addToAgentsInRegion(agent);
				}
			}
		}

		void removeAgentFromRegion(Tagent* agent) {
			for(auto region : regions) {
				if (region->contains(agent->getX(), agent->getY())) {
					region->removeFromAgentsInRegion(agent);
				}
			}
		}

		Region* getAgentCurrentRegion(Tagent* agent) const {
			for(auto region : regions) {
				if (region->contains(agent->getX(), agent->getY())) {
					return region;
				}
			}

			return NULL;
		}

	private:

		// Denotes which implementation (sequential, parallel implementations..)
		// should be used for calculating the desired positions of
		// agents (Assignment 1)
		IMPLEMENTATION implementation;

		// The agents in this scenario
		std::vector<Tagent*> agents;

		// For SoA-based (SIMD) implementation:
    	AgentsSoA agentsSoA;

		// For SoA-based (SIMD) implementation:
		WaypointsSoA waypointsSoA;

		// The waypoints in this scenario
		std::vector<Twaypoint*> destinations;

		// Moves an agent towards its next position
		void move(Ped::Tagent *agent);

		////////////
		/// Everything below here won't be relevant until Assignment 3
		///////////////////////////////////////////////

		std::vector<Region*> regions;

		// Returns the set of neighboring agents for the specified position
		set<const Ped::Tagent*> getNeighbors(int x, int y, int dist) const;

		////////////
		/// Everything below here won't be relevant until Assignment 4
		///////////////////////////////////////////////

#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE SIZE*CELLSIZE

		// The heatmap representing the density of agents
		int ** heatmap;

		// The scaled heatmap that fits to the view
		int ** scaled_heatmap;

		// The final heatmap: blurred and scaled to fit the view
		int ** blurred_heatmap;

		void setupHeatmapSeq();
		void freeHeatmapMemory();
		void updateHeatmapSeq();
	};
}
#endif
