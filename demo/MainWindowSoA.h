#ifndef MAINWINDOW_SOA_H
#define MAINWINDOW_SOA_H

#include <QMainWindow>
#include <QGraphicsScene>
#include <vector>

#include "ped_model.h"
#include "ped_agent.h"
#include "ViewAgentSoA.h"

class MainWindowSoA : public QMainWindow {
    Q_OBJECT

public:
    MainWindowSoA(const Ped::Model &pedModel);
    void paint();
    ~MainWindowSoA();

    static int cellToPixel(int val);
    static constexpr int cellsizePixel = 10;

private:
    const Ped::Model &model;
    QGraphicsView *graphicsView;
    QGraphicsScene *scene;
    QGraphicsPixmapItem *pixmap;
    std::vector<ViewAgentSoA*> viewAgentsSoA;
};

#endif // MAINWINDOW_SOA_H
