#pragma once

#include <pcl/visualization/pcl_visualizer.h>
#include "Common.h"

class PclVisualization {

    pcl::visualization::PCLVisualizer visu;

public:

    PclVisualization();

    void addCamera(cv::Ptr<Camera> cam, const cv::Matx44d &P, long frameId);

    void addPointCloud(const std::vector<CloudPoint> &points, long frameId);

    void spin() { visu.spin(); }

    void stop() { visu.spinOnce(); }

    void init();

    void addChessboard();
};
