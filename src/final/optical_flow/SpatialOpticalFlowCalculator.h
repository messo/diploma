#pragma once

#include <opencv2/core/mat.hpp>
#include "../camera/Camera.hpp"
#include "OFReconstruction.h"
#include "OpticalFlowCalculator.h"

class SpatialOpticalFlowCalculator : public OpticalFlowCalculator {

public:

    SpatialOpticalFlowCalculator(cv::Ptr<Camera> camera1, cv::Ptr<Camera> camera2) :
            OpticalFlowCalculator(camera1, camera2) { }

    bool feed(const cv::Mat frame1, const cv::Mat mask1,
              const cv::Mat frame2, const cv::Mat mask2);

};