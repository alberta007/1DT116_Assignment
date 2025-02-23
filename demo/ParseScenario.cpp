//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017.
// Modified in 2025 to remove QT's XML parser and used TinyXML2 instead.


#include "ParseScenario.h"
#include <string>
#include <iostream>

#include <stdlib.h>
#include <soa_agent.h>
#include <waypoints_soa.h>

// Comparator used to identify if two agents differ in their position
bool positionComparator(Ped::Tagent *a, Ped::Tagent *b) {
	// True if positions of agents differ
	return (a->
	getX() < b->getX()) || ((a->getX() == b->getX()) && (a->getY() < b->getY()));
}

const AgentsSoA ParseScenario::getAgentsSoA() const {
    return agentsSoA; 
}

// Reads in the configuration file, given the filename
ParseScenario::ParseScenario(std::string filename, bool verbose)
{
	XMLError ret = doc.LoadFile(filename.c_str());
	if (ret != XML_SUCCESS) {
		//std::cout << "Error reading the scenario configuration file: " << ret << std::endl;
		fprintf(stderr, "Error reading the scenario configuration file for filename %s: ", filename.c_str());
		perror(NULL);
		exit(1);
		return;
	}

	// Get the root element (welcome)
	XMLElement* root = doc.FirstChildElement("welcome");
	if (!root) {
		std::cerr << "Error: Missing <welcome> element in the XML file!" << std::endl;
		exit(1);
		return;
	}

	// Parse waypoints
	if (verbose) std::cout << "Waypoints:" << std::endl;
	for (XMLElement* waypoint = root->FirstChildElement("waypoint"); waypoint; waypoint = waypoint->NextSiblingElement("waypoint")) {
		std::string id = waypoint->Attribute("id");
		double x = waypoint->DoubleAttribute("x");
		double y = waypoint->DoubleAttribute("y");
		double r = waypoint->DoubleAttribute("r");

		if (verbose) std::cout << "  ID: " << id << ", x: " << x << ", y: " << y << ", r: " << r << std::endl;

		Ped::Twaypoint *w = new Ped::Twaypoint(x, y, r);
		waypoints[id] = w;

		// Load the waypoint into the SoA representation
		int index = waypointsSoA.x.size();  // next free index in SoA
		waypointsSoA.x.push_back((float)x);
		waypointsSoA.y.push_back((float)y);
		waypointsSoA.r.push_back((float)r);

		// Keep track of the index for this ID
		waypointIndexById[id] = index;
	}

	size_t oldSizeSoA = agentsSoA.x.size();

if (verbose) std::cout << "\nAgents:" << std::endl;
for (XMLElement* agent = root->FirstChildElement("agent"); agent; agent = agent->NextSiblingElement("agent")) {
    double x = agent->DoubleAttribute("x");
    double y = agent->DoubleAttribute("y");
    int n    = agent->IntAttribute("n");
    double dx = agent->DoubleAttribute("dx");
    double dy = agent->DoubleAttribute("dy");

    if (verbose) {
        std::cout << "  Agent: x: " << x 
                  << ", y: " << y 
                  << ", n: " << n
                  << ", dx: " << dx 
                  << ", dy: " << dy 
                  << std::endl;
    }

    // We'll extend agentsSoA by 'n' for these new agents
    size_t startIndex = agentsSoA.x.size();
    agentsSoA.resize(startIndex + n);

    // This temp vector is for the old Tagent approach:
    tempAgents.clear();

    // Create the Tagent objects AND fill the SoA positions
    std::set<std::pair<int, int>> positions;
    for (int i = 0; i < n; ++i) {
        int xPos, yPos;
        do {
            xPos = (int)(x + rand() / (RAND_MAX / dx) - dx / 2);
            yPos = (int)(y + rand() / (RAND_MAX / dy) - dy / 2);
        } while (positions.find({xPos, yPos}) != positions.end());

        positions.insert({xPos, yPos});

        // create a Tagent
        Ped::Tagent* a = new Ped::Tagent(xPos, yPos);
        tempAgents.push_back(a);

        // Also fill SoA for agent i of this block
        size_t idx = startIndex + i;
        agentsSoA.x[idx]        = xPos;
        agentsSoA.y[idx]        = yPos;

		
        agentsSoA.currentWaypointIndex[idx] = 0; 
    }

    bool firstWaypointUsed = false;
    for (XMLElement* addwaypoint = agent->FirstChildElement("addwaypoint"); 
         addwaypoint; 
         addwaypoint = addwaypoint->NextSiblingElement("addwaypoint"))
    {
        std::string id = addwaypoint->Attribute("id");
        if (verbose) {
            std::cout << "    AddWaypoint ID: " << id << std::endl;
        }
        for (auto a : tempAgents) {
            a->addWaypoint(waypoints[id]);
        }

		int waypointIndex = waypointIndexById[id];

        if (!firstWaypointUsed) {
            // get the numeric coords from the Twaypoint pointer
            Ped::Twaypoint* wp = waypoints[id];
            float wpX = (float)wp->getx();
            float wpY = (float)wp->gety();
            float wpR = (float)wp->getr();

            // For each new agent in SoA, set destX/Y
            for (int i = 0; i < n; i++) {
                size_t idx = startIndex + i;
				agentsSoA.currentWaypointIndex[idx] = waypointIndex;
                agentsSoA.destX[idx] = wpX;
                agentsSoA.destY[idx] = wpY;
                agentsSoA.destR[idx] = wpR;
                
            }
            firstWaypointUsed = true;
        }
       
    }

    // Insert these Tagent objects into the old 'agents' vector
    agents.insert(agents.end(), tempAgents.begin(), tempAgents.end());
}
tempAgents.clear();

	// Hack! Do not allow agents to be on the same position. Remove duplicates from scenario and free the memory.
	bool(*fn_pt)(Ped::Tagent*, Ped::Tagent*) = positionComparator;
	std::set<Ped::Tagent*, bool(*)(Ped::Tagent*, Ped::Tagent*)> agentsWithUniquePosition(fn_pt);
	int duplicates = 0;
	for (auto agent : agents)
	{
		if (agentsWithUniquePosition.find(agent) == agentsWithUniquePosition.end())
		{
			agentsWithUniquePosition.insert(agent);
		}
		else
		{
			delete agent;
			duplicates += 1;
		}
	}
	if (duplicates > 0)
	{
		std::cout << "Note: removed " << duplicates << " duplicates from scenario." << std::endl;
	}
	agents = std::vector<Ped::Tagent*>(agentsWithUniquePosition.begin(), agentsWithUniquePosition.end());
}

vector<Ped::Tagent*> ParseScenario::getAgents() const
{
	return agents;
}

std::vector<Ped::Twaypoint*> ParseScenario::getWaypoints()
{
	std::vector<Ped::Twaypoint*> v; //
	for (auto p : waypoints)
	{
		v.push_back((p.second));
	}
	return std::move(v);
}


// Used to free all dynamically allocated memory
ParseScenario::~ParseScenario() {
    // Free all dynamically allocated agents
    for (auto agent : agents) {
        delete agent;
    }
    agents.clear();

    // Free all dynamically allocated waypoints
    for (auto waypointPair : waypoints) {
        delete waypointPair.second;
    }
    waypoints.clear();
}
