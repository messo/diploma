#pragma once

#include <opencv2/core/mat.hpp>
#include "../Object.h"
#include "../camera/Camera.hpp"
#include "OFReconstruction.h"
#include "OpticalFlowCalculator.h"

class MultiObjectOpticalFlowCalculator : public OpticalFlowCalculator {

public:

    MultiObjectOpticalFlowCalculator(cv::Ptr<Camera> camera1, cv::Ptr<Camera> camera2, cv::Mat F) :
            OpticalFlowCalculator(camera1, camera2, F) { }

    bool feed(std::vector<cv::Mat> &frames, const Object &object);

protected:
    virtual double calcOpticalFlow(cv::Point &translation) override;

    virtual void collectMatchingPoints(const cv::Mat &flow, const cv::Mat &backFlow, const cv::Rect &roi,
                                       std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2) override;

};
