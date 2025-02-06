#include "MainWindowSoA.h"

#include <QGraphicsView>
#include <QtGui>
#include <QBrush>

#include <iostream>

#include <stdlib.h>

MainWindowSoA::MainWindowSoA(const Ped::Model &pedModel) : model(pedModel) {
    graphicsView = new QGraphicsView();
    setCentralWidget(graphicsView);
    scene = new QGraphicsScene(QRect(0, 0, 800, 600), this);
    graphicsView->setScene(scene);
    scene->setBackgroundBrush(Qt::white);

    // Draw grid lines
    for (int x = 0; x <= 800; x += cellsizePixel) {
        scene->addLine(x, 0, x, 600, QPen(Qt::gray));
    }
    for (int y = 0; y <= 600; y += cellsizePixel) {
        scene->addLine(0, y, 800, y, QPen(Qt::gray));
    }

    // Create view agents using SoA
    const AgentsSoA &agentsSoA = model.getAgentsSoA();
    size_t num_agents = agentsSoA.x.size();

    for (size_t i = 0; i < num_agents; i++) {
        viewAgentsSoA.push_back(new ViewAgentSoA(scene, i, agentsSoA));
    }

    // Setup heatmap
    const int heatmapSize = model.getHeatmapSize();
    QPixmap pixmapDummy = QPixmap(heatmapSize, heatmapSize);
    pixmap = scene->addPixmap(pixmapDummy);

    paint();
    graphicsView->show();
}

void MainWindowSoA::paint() {
    std::set<std::tuple<int, int>> positionsTaken;
    for (auto &viewAgent : viewAgentsSoA) {
        size_t beforeInsert = positionsTaken.size();
        positionsTaken.insert(viewAgent->getPosition());
        size_t afterInsert = positionsTaken.size();

        QColor color = (beforeInsert != afterInsert) ? Qt::green : Qt::red;
        viewAgent->paint(color);
    }
}

int MainWindowSoA::cellToPixel(int val) {
    return val * cellsizePixel;
}

MainWindowSoA::~MainWindowSoA() {
    for (auto agent : viewAgentsSoA) {
        delete agent;
    }
}
