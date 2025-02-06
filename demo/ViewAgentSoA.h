#ifndef VIEWAGENT_SOA_H
#define VIEWAGENT_SOA_H

#include <QGraphicsScene>
#include <QGraphicsRectItem>
#include <QColor>
#include "ped_model.h"

class ViewAgentSoA {
public:
    ViewAgentSoA(QGraphicsScene *scene, size_t index, const AgentsSoA &agentsSoA);
    void paint(QColor color);
    std::pair<int, int> getPosition() const;

private:
    size_t index;  // Stores the agent index in the SoA structure
    const AgentsSoA &agentsSoA;  // Reference to the SoA agents
    QGraphicsRectItem *rect;
};

#endif // VIEWAGENT_SOA_H
