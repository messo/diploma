/*
 * Camera.h
 *
 *  Created on: 2014.03.08.
 *      Author: Balint
 */

#ifndef CAMERA_H_
#define CAMERA_H_

#define _USE_MATH_DEFINES

#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace cv;

class Camera {
    Mat rvec, tvec;
public:
    Camera();

    virtual ~Camera();

    void setRotation(int rot1, int rot2, int rot3);

    void rotX(int deg);

    void rotY(int deg);

    void rotZ(int deg);

    void setTranslation(double x, double y, double z);

    void addX(double x);

    void addY(double x);

    void addZ(double x);

    const Mat &getRVec() const {
        return rvec;
    }

    const Mat &getTVec() const {
        return tvec;
    }
};

#endif /* CAMERA_H_ */
