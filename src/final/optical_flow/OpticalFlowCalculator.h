#pragma once

#include <opencv2/core/mat.hpp>
#include "../camera/Camera.hpp"
#include "OFReconstruction.h"

class OpticalFlowCalculator {

protected:

    double EPSILON_TEXTURE = 6.0;

    cv::Ptr<Camera> camera1;
    cv::Ptr<Camera> camera2;

    cv::Mat frames[2];
    cv::Mat masks[2];
    cv::Mat texturedRegions[2];

    void calcTexturedRegions(const cv::Mat frame, const cv::Mat mask, cv::Mat &texturedRegions) const;

    cv::Point2f calcAverageMovement(const std::vector<cv::Point2f> &points1,
                                    const std::vector<cv::Point2f> &points2) const;

    void collectMatchingPoints(const cv::Mat &flow, const cv::Mat &backFlow, const cv::Rect &roi,
                               std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2);

    /** VISUALIZATIONS */

    void visualizeOpticalFlow(const cv::Mat &img1, const cv::Mat &mask1,
                              const cv::Mat &img2, const cv::Mat &mask2,
                              const cv::Mat &flow, const std::string &name) const;

    void visualizeMatches(const std::vector<cv::Point2f> &points1, const std::vector<cv::Point2f> &points2) const;

    void visualizeMatches(const cv::Mat &img1, const std::vector<cv::Point2f> &points1,
                          const cv::Mat &img2, const std::vector<cv::Point2f> &points2) const;

    OpticalFlowCalculator(cv::Ptr<Camera> camera1, cv::Ptr<Camera> camera2) : camera1(camera1), camera2(camera2) { }

    double calcOpticalFlow(cv::Point &translation);

public:
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;

};
