#pragma once

#include <string>
#include <iostream>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "StereoCamera.hpp"

using namespace cv;
using namespace std;

class StereoCamera;

class Calibration {

    vector<Mat> images;
    Size imageSize;

public:
    Mat cameraMatrix[2], distCoeffs[2];
    Mat R, T, Q;
    Mat rmap[2][2];
    Mat P1, P2, E, F;
    Rect validRoiLeft, validRoiRight;

    Calibration() {
    };

    Calibration(const string &intrinsics, const string &extrinsics);

    void acquireFrames(StereoCamera &stereoCamera);

    void calibrate();
};
