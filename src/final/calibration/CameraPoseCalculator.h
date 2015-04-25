#pragma once

#include <opencv2/core/cvstd.hpp>
#include "../camera/CameraPose.h"
#include "../camera/Camera.hpp"

class CameraPoseCalculator {

    cv::Ptr<Camera> camera;

public:

    cv::Ptr<CameraPose> cameraPose;

    std::vector<cv::Point2f> imagePoints;

    CameraPoseCalculator(cv::Ptr<Camera> camera) : camera(camera), cameraPose(new CameraPose()) { }

    bool calculate();

    bool poseCalculated();

};
