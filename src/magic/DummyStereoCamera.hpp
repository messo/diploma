#pragma once

#include <iostream>
#include "StereoCamera.hpp"

class DummyStereoCamera : public StereoCamera {

    int frameId;

    Mat getImage(int cam);

public:
    DummyStereoCamera(Calibration *calibration = NULL) : StereoCamera(calibration), frameId(-1) {
    }

    virtual Mat getLeft();

    virtual Mat getRight();
};
