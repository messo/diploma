#pragma once

#include <opencv2/core/mat.hpp>
#include "../camera/Camera.hpp"
#include "OFReconstruction.h"
#include "OpticalFlowCalculator.h"
#include "../ObjectSelector.hpp"

class SpatialOpticalFlowCalculator : public OpticalFlowCalculator {

public:

    SpatialOpticalFlowCalculator(cv::Ptr<Camera> camera1, cv::Ptr<Camera> camera2) :
            OpticalFlowCalculator(camera1, camera2) { }

    bool feed(cv::Mat (&frames)[2], ObjectSelector (&objSelector)[2]);

};
