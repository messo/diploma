#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "Calibration.hpp"

using namespace cv;

class Calibration;

class StereoCamera {
protected:
    Size imageSize;
    Calibration *calibration;
    Rect dispRoi;
    Ptr<StereoSGBM> sgbm;

    Mat rectify(const Mat &img, int cam);

public:
    StereoCamera(Calibration *calibration = NULL);

    virtual Mat getLeft() = 0;

    virtual Mat getRight() = 0;

    Mat getDisparityMatrix(Mat &left, Mat &right);

    Mat normalizeDisparity(Mat const &imgDisparity16S);
};
