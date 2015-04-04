#pragma once

#include <opencv2/core/mat.hpp>
#include "../camera/Camera.hpp"
#include "OpticalFlowCalculator.h"
#include "OFReconstruction.h"

class TemporalOpticalFlowCalculator : public OpticalFlowCalculator {

    cv::Rect currentBoundingRect;

    std::vector<cv::Point> prevContour;
    std::vector<cv::Point> currentContour;

public:

    long prevFrameId;
    long currentFrameId;

    TemporalOpticalFlowCalculator(cv::Ptr<Camera> camera) : OpticalFlowCalculator(camera, camera) { }

    bool feed(long frameId, const cv::Mat &image, const cv::Mat &lastMask, const cv::Rect &boundingRect,
              const std::vector<cv::Point> &vector);

};
