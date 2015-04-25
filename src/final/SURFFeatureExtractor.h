#pragma once

#include <opencv2/xfeatures2d/nonfree.hpp>

class SURFFeatureExtractor {

    cv::Ptr<cv::xfeatures2d::SURF> extractor;

public:
    std::vector<std::vector<cv::KeyPoint>> keypoints;
    std::vector<cv::Mat> descriptors;

    SURFFeatureExtractor(const std::vector<cv::Mat> &images, const std::vector<cv::Mat> &masks);
};
