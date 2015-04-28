#pragma once

#include <string>
#include <iostream>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../camera/StereoCamera.hpp"

class StereoCamera;

class StereoCalibration {

    std::vector<cv::Mat> images;
    cv::Size imageSize;

public:
    cv::Mat cameraMatrix[2], distCoeffs[2];
    cv::Mat R, T, Q;
    cv::Mat rmap[2][2];
    cv::Mat P1, P2, E, F;
    cv::Rect validRoiLeft, validRoiRight;

    StereoCalibration() {
    };

    StereoCalibration(const std::string &intrinsics, const std::string &extrinsics);

    void acquireFrames(StereoCamera &stereoCamera);

    void calibrate();
};
