#pragma once

#include <opencv2/core/mat.hpp>
#include "Camera.hpp"
#include "OFReconstruction.h"

class OpticalFlowCalculator {

    cv::Ptr<Camera> camera;

    cv::Mat prevFrame;
    std::vector<cv::Point> prevContour;

    cv::Mat currentFrame;
    std::vector<cv::Point> currentContour;

    cv::Ptr<OFReconstruction> currentReconstruction;

    void visualizeOpticalFlow(const cv::Mat &flow);

    void visualizeMatches(const std::vector<cv::Point2f> &points1, const std::vector<cv::Point2f> &points2);

public:

    OpticalFlowCalculator(cv::Ptr<Camera> camera) : camera(camera), currentFrame(480, 640, CV_8U) {

    }

    void feed(const cv::Mat &image, const cv::Rect &boundingRect, const std::vector<cv::Point> &vector);

    void calcOpticalFlow();

    void collectMatchingPoints(const cv::Mat &flow, const cv::Mat &backFlow, std::vector<cv::Point2f> &imgpts1, std::vector<cv::Point2f> &imgpts2);

    void dumpReconstruction();
};
