#pragma once

#include <opencv2/core.hpp>
#include "../camera/Camera.hpp"
#include "../Common.h"

class OFReconstruction {

    cv::Ptr<Camera> cam;
    long frameId1;
    std::vector<cv::Point2f> pts1;
    long frameId2;
    std::vector<cv::Point2f> pts2;

public:

    cv::Matx34d P1, P2;
    Cloud resultingCloud;

    OFReconstruction(cv::Ptr<Camera> camera, long frameId1, std::vector<cv::Point2f> pts1, long frameId2,
                     std::vector<cv::Point2f> pts2);

    bool reconstruct();

};
