#pragma once


#include "Blob.h"

class Object {


public:
    std::vector<cv::Mat> masks;

    std::vector<std::pair<cv::Point2f, cv::Point2f>> matches;

    Object(cv::Mat left, cv::Mat right);
};
