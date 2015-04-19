#pragma once

#include <iostream>
#include "camera/CameraPose.h"
#include "Common.h"
#include "locking.h"

class MatVisualization {

    CameraPose *cameraPose;
    cv::Mat cameraMatrix;
    cv::Mat result;

    MutexType mutexType;

public:

    MatVisualization(CameraPose &pose, const cv::Mat &cameraMatrix) :
            result(480, 640, CV_8UC3, cv::Scalar(0, 0, 0)), cameraPose(&pose), cameraMatrix(cameraMatrix) { }

    void renderPointCloud(const std::vector<CloudPoint> &points);

    cv::Mat getResult();
};



