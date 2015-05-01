#pragma once

#include <opencv2/core.hpp>
#include "../camera/Camera.hpp"

class Matcher {

    cv::Ptr<Camera> camera1;
    cv::Ptr<Camera> camera2;
    cv::Mat F;

    std::vector<std::vector<cv::KeyPoint>> keypoints;
    std::vector<cv::Mat> descriptors;

    double threshold = 0.2;

    void detectKeypointsAndExtractDescriptors(const std::vector<cv::Mat> &images, const std::vector<cv::Mat> &masks);

    std::vector<cv::DMatch> matchDescriptors();

    std::vector<std::pair<cv::Point2f, cv::Point2f>> buildMatches(const std::vector<cv::DMatch> &matches,
                                                                  std::vector<cv::DMatch> &keptMatches);

public:

    Matcher(cv::Ptr<Camera> cam1, cv::Ptr<Camera> cam2, cv::Mat F) : camera1(cam1), camera2(cam2), F(F) { }

    std::vector<std::pair<cv::Point2f, cv::Point2f>> match(const std::vector<cv::Mat> &images,
                                                           const std::vector<cv::Mat> &masks);
};
