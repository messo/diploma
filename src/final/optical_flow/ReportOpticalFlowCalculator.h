#pragma once

#include <opencv2/core/mat.hpp>
#include "../camera/Camera.hpp"
#include "OpticalFlowCalculator.h"

class ReportOpticalFlowCalculator : public OpticalFlowCalculator {

public:

    virtual std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> calcDenseMatches(std::vector<cv::Mat> &frames, const Object &object) override;

protected:
    virtual std::vector<cv::Mat> calcOpticalFlows() const override;

    virtual std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> collectMatchingPoints(const std::vector<cv::Mat> &flows) const override;

};
