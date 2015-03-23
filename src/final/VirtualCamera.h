#pragma once
#define _USE_MATH_DEFINES

#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

class VirtualCamera {
    cv::Mat rvec, tvec;
public:
    VirtualCamera();

    virtual ~VirtualCamera();

    void setRotation(int rot1, int rot2, int rot3);

    void rotX(int deg);

    void rotY(int deg);

    void rotZ(int deg);

    void setTranslation(double x, double y, double z);

    void addX(double x);

    void addY(double x);

    void addZ(double x);

    const cv::Mat &getRVec() const {
        return rvec;
    }

    const cv::Mat &getTVec() const {
        return tvec;
    }
};
