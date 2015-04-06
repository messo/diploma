#pragma once

#include <opencv2/core/cvstd.hpp>
#include "../camera/CameraPose.h"
#include "../camera/Camera.hpp"

class CameraPoseCalculator {

    cv::Ptr<Camera> camera;

public:

    cv::Ptr<CameraPose> cameraPose;

    CameraPoseCalculator(cv::Ptr<Camera> camera) : camera(camera), cameraPose(new CameraPose()) { }

    bool calculate();

    bool poseCalculated();

};
