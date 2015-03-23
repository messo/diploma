#pragma once

#include <iostream>
#include <opencv2/core/types.hpp>

struct CloudPoint {
    cv::Point3d pt;
    std::vector<int> imgpt_for_img;
    double reprojection_error;
};

std::vector<cv::Point3d> CloudPointsToPoints(const std::vector<CloudPoint> &cpts);

void writeCloudPoints(const std::vector<CloudPoint> &cpts);

void translate(const std::vector<cv::Point>& input, cv::Point translation, std::vector<cv::Point>& output);

cv::Mat mergeImages(const cv::Mat &left, const cv::Mat &right);
