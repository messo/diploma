#pragma once

#include <opencv2/core/mat.hpp>
#include "camera/Camera.hpp"
#include "OFReconstruction.h"

class OpticalFlowCalculator {

    cv::Ptr<Camera> camera;

    cv::Mat prevFrame;
    cv::Mat prevMask;
    cv::Mat prevTexturedRegions;
    std::vector<cv::Point> prevContour;

    cv::Rect currentBoundingRect;
    cv::Mat currentMask;
    cv::Mat currentTexturedRegions;
    std::vector<cv::Point> currentContour;

    void visualizeOpticalFlow(const cv::Mat &flow);

    void visualizeMatches(const std::vector<cv::Point2f> &points1, const std::vector<cv::Point2f> &points2);

public:

    long prevFrameId;
    std::vector<cv::Point2f> points1;
    long currentFrameId;
    std::vector<cv::Point2f> points2;

    cv::Mat currentFrame;

    OpticalFlowCalculator(cv::Ptr<Camera> camera) : camera(camera), currentFrame(480, 640, CV_8U),
                                                    currentMask(480, 640, CV_8U),
                                                    currentTexturedRegions(480, 640, CV_8U) {
    }

    bool feed(long frameId, const cv::Mat &image, const cv::Mat &lastMask, const cv::Rect &boundingRect,
              const std::vector<cv::Point> &vector);

    double calcOpticalFlow();

    void collectMatchingPoints(const cv::Mat &flow, const cv::Mat &backFlow, std::vector<cv::Point2f> &imgpts1,
                               std::vector<cv::Point2f> &imgpts2);

    void calcTexturedRegions();

    cv::Point2f calcAverageMovement(const std::vector<cv::Point2f> &points1, const std::vector<cv::Point2f> &points2);
};
