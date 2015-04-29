#pragma once

#include <opencv2/core.hpp>
#include "../camera/Camera.hpp"
#include "OpticalFlowCalculator.h"
#include "../object/Object.h"

class SpatialOpticalFlowCalculator : public OpticalFlowCalculator {

public:

    SpatialOpticalFlowCalculator(cv::Ptr<Camera> camera1, cv::Ptr<Camera> camera2, cv::Mat F) :
            OpticalFlowCalculator(camera1, camera2, F) { }

    bool feed(std::vector<cv::Mat> &frames, const Object &object);

};
