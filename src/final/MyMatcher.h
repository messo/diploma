#pragma once


#include <opencv2/core/mat.hpp>
#include "SURFFeatureExtractor.h"
#include "camera/Camera.hpp"

class MyMatcher {

    cv::Ptr<Camera> camera1;
    cv::Ptr<Camera> camera2;

    double ratio = 0.65;
    double treshold = 0.3;

    int ratioTest(std::vector<std::vector<cv::DMatch>> vector);

    void symmetryTest(const std::vector<std::vector<cv::DMatch>> &matches1,
                      const std::vector<std::vector<cv::DMatch>> &matches2,
                      std::vector<cv::DMatch> &symmMatches);

    std::vector<std::pair<cv::Point2f, cv::Point2f>> filterWithF(const std::vector<std::vector<cv::KeyPoint>> &keypoints,
                     const std::vector<cv::DMatch> &before,
                     std::vector<cv::DMatch> &after,
                     cv::Mat F);

public:
    MyMatcher(cv::Ptr<Camera> camera1, cv::Ptr<Camera> camera2) : camera1(camera1), camera2(camera2) { }

    std::vector<std::pair<cv::Point2f, cv::Point2f>> match(SURFFeatureExtractor extractor, std::vector<cv::DMatch> &matches, cv::Mat F);
};
