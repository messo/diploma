#pragma once

#include <opencv2/core/mat.hpp>
#include "../camera/Camera.hpp"
#include "OFReconstruction.h"
#include "OpticalFlowCalculator.h"

class ReportOpticalFlowCalculator : public OpticalFlowCalculator {

public:

    ReportOpticalFlowCalculator(cv::Ptr<Camera> camera1, cv::Ptr<Camera> camera2) :
            OpticalFlowCalculator(camera1, camera2) { }

    bool feed(std::vector<cv::Mat> &frames, std::vector<cv::Mat> &masks);

protected:
    virtual double calcOpticalFlow(cv::Point &translation) override;

    virtual void collectMatchingPoints(const cv::Mat &flow, const cv::Mat &backFlow, const cv::Rect &roi,
                                       std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2) override;
};
