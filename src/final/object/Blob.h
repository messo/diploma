#pragma once

#include <opencv2/core.hpp>

struct Blob {

    cv::Mat mask;

    Blob(cv::Size imgSize, std::vector<cv::Point> &contour);

    bool contains(const cv::Point2f& point) const;

};
