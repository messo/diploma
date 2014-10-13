#pragma once

#include <iostream>
#include "StereoCamera.hpp"

class RealStereoCamera : public StereoCamera {

    VideoCapture leftCamera, rightCamera;

    Mat getImage(VideoCapture& capture, int cam);

public:
    RealStereoCamera(Calibration *calibration = NULL) : StereoCamera(calibration) {
        if (!leftCamera.open(0)) {
            std::cout << "Cannot find left camera!" << std::endl;
        }
        if (!rightCamera.open(1)) {
            std::cout << "Cannot find right camera!" << std::endl;
        }
    }

    virtual Mat getLeft();

    virtual Mat getRight();
};
