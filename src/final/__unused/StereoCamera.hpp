#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "../calibration/StereoCalibration.hpp"
#include "Camera.hpp"

class StereoCalibration;

class StereoCamera {
    int counter = -1;
    time_t start, end;
    double fps;

    cv::Ptr<Camera> leftCamera, rightCamera;

    cv::Size imageSize;

    cv::Mat getImage(cv::Ptr<Camera> camera);

    cv::Mat rectify(const cv::Mat &img, int cam);

public:

    cv::Ptr<StereoCalibration> calibration;
    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;
    cv::Rect dispRoi;

    StereoCamera(cv::Ptr<StereoCalibration> calibration = cv::Ptr<StereoCalibration>());

    StereoCamera(std::string path, int count, cv::Ptr<StereoCalibration> calibration = cv::Ptr<StereoCalibration>());

    cv::Mat getLeft();

    cv::Mat getRight();

    cv::Mat getDisparityMatrix(const cv::Mat &left, const cv::Mat &right);

    cv::Mat normalizeDisparity(cv::Mat const &disparity16S);

    bool reprojectTo3D(const cv::Mat &disparity16S);

    void getCameraPose(cv::OutputArray rvec, cv::OutputArray tvec);

    void reprojectPoints(const cv::Mat &rvec, const cv::Mat &tvec, const cv::Mat &img, cv::Mat &output);

    void initCalibration();
};
