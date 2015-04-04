#pragma once

#include <opencv2/core/mat.hpp>
#include "../camera/Camera.hpp"
#include "OFReconstruction.h"

class OpticalFlowCalculator {

protected:

    double EPSILON_TEXTURE = 6.0;

    cv::Ptr<Camera> camera1;
    cv::Ptr<Camera> camera2;

    cv::Mat frame1;
    cv::Mat mask1;
    cv::Mat texturedRegions1;

    cv::Mat frame2;
    cv::Mat mask2;
    cv::Mat texturedRegions2;

    void calcTexturedRegions(const cv::Mat frame, const cv::Mat mask, cv::Mat& texturedRegions) const;

    cv::Point2f calcAverageMovement(const std::vector<cv::Point2f> &points1, const std::vector<cv::Point2f> &points2) const;

    void collectMatchingPoints(const cv::Mat &flow, const cv::Mat &backFlow,
                               std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2);

    /** VISUALIZATIONS */

    void visualizeOpticalFlow(const cv::Mat &flow) const;

    void visualizeMatches(const std::vector<cv::Point2f> &points1, const std::vector<cv::Point2f> &points2) const;

    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;

    OpticalFlowCalculator(cv::Ptr<Camera> camera1, cv::Ptr<Camera> camera2) : camera1(camera1), camera2(camera2) { }

    double calcOpticalFlow();
};
