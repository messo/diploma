#pragma once


#include <opencv2/core/mat.hpp>

class ObjectSelector {

    cv::Ptr<cv::Point> getCentroid(cv::Moments moments);

    std::vector<cv::Point> lastContour;

public:
    ObjectSelector();

    cv::Mat selectUsingConnectedComponents(const cv::Mat &img, const cv::Mat &mask);

    cv::Mat selectUsingContoursWithMaxArea(const cv::Mat &img, const cv::Mat &mask);

    cv::Mat selectUsingContoursWithClosestCentroid(const cv::Mat &img, const cv::Mat &mask);

    const std::vector<cv::Point> &getLastContour() const;


    cv::Ptr<cv::Point> lastCentroid;

    cv::Rect lastBoundingRect;

    cv::Mat lastMask;
};
