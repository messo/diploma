#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "Calibration.hpp"
#include "Camera.hpp"

class Calibration;

class StereoCamera {
    int counter = -1;
    time_t start, end;
    double fps;

    cv::Ptr<Camera> leftCamera, rightCamera;

    cv::Size imageSize;
    cv::Ptr<Calibration> calibration;


    cv::Mat getImage(cv::Ptr<Camera> camera);

    cv::Mat rectify(const cv::Mat &img, int cam);

public:
    enum Type {
        REAL, DUMMY
    };


    cv::Rect dispRoi;

    StereoCamera(Type type, cv::Ptr<Calibration> calibration = cv::Ptr<Calibration>());

    cv::Mat getLeft();

    cv::Mat getRight();

    cv::Mat getDisparityMatrix(const cv::Mat &left, const cv::Mat &right);

    cv::Mat normalizeDisparity(cv::Mat const &imgDisparity16S);
};
