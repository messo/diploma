#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "Calibration.hpp"

using namespace cv;

class Calibration;

class StereoCamera {
protected:
    Calibration *calibration;

    Mat rectify(const Mat &img, int cam);

public:
    StereoCamera(Calibration *calibration = NULL) :
            calibration(calibration) {
    };

    virtual Mat getLeft() = 0;

    virtual Mat getRight() = 0;
};
