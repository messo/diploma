#pragma once

#include <iostream>
#include "camera/CameraPose.h"
#include "Common.h"
#include "locking.h"

class Visualization {

    const double REPROJ_ERROR_THRESHOLD = 16.0;

    const CameraPose &cameraPose;
    const cv::Mat &cameraMatrix;

    cv::Mat result;

    MutexType mutexType;

public:

    Visualization(const CameraPose &pose, const cv::Mat &cameraMatrix) :
            result(480, 640, CV_8UC3, cv::Scalar(0, 0, 0)), cameraPose(pose), cameraMatrix(cameraMatrix) {
    }

    void renderWithDepth(const std::vector<CloudPoint> &points);

    void renderWithContours(const std::vector<CloudPoint> &points);

    void renderWithContours(const std::vector<std::vector<CloudPoint>> &points);

    void renderWithColors(const std::vector<CloudPoint> &points,
                          const std::vector<cv::Point2f> &originalPoints,
                          const cv::Mat &image);

    void renderWithGrayscale(const std::vector<CloudPoint> &points,
                             const std::vector<cv::Point2f> &originalPoints,
                             const cv::Mat &image);

    cv::Mat getResult();
};
