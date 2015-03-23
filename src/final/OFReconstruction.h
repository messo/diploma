#pragma once

#include <bits/stl_bvector.h>
#include <opencv2/core/types.hpp>
#include "Camera.hpp"

class OFReconstruction {

    cv::Ptr<Camera> cam;
    std::vector<cv::Point2f> pts1;
    std::vector<cv::Point2f> pts2;

public:
    OFReconstruction(cv::Ptr<Camera> pts1, std::vector<cv::Point2f> pts2, std::vector<cv::Point2f> vector);

    bool reconstruct();
};