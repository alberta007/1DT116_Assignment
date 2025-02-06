#include "ViewAgentSoA.h"
#include "MainWindowSoA.h"

ViewAgentSoA::ViewAgentSoA(QGraphicsScene *scene, size_t index_, const AgentsSoA &agentsSoA_) 
    : index(index_), agentsSoA(agentsSoA_)
{
    QBrush greenBrush(Qt::green);
    QPen outlinePen(Qt::black);
    outlinePen.setWidth(2);

    rect = scene->addRect(MainWindowSoA::cellToPixel(agentsSoA.x[index]), 
                          MainWindowSoA::cellToPixel(agentsSoA.y[index]),
                          MainWindowSoA::cellsizePixel - 1, 
                          MainWindowSoA::cellsizePixel - 1, outlinePen, greenBrush);
}

void ViewAgentSoA::paint(QColor color) {
    QBrush brush(color);
    rect->setBrush(brush);
    rect->setRect(MainWindowSoA::cellToPixel(agentsSoA.x[index]), 
                  MainWindowSoA::cellToPixel(agentsSoA.y[index]),
                  MainWindowSoA::cellsizePixel - 1, 
                  MainWindowSoA::cellsizePixel - 1);
}

std::pair<int, int> ViewAgentSoA::getPosition() const {
    return {agentsSoA.x[index], agentsSoA.y[index]};
}
