#pragma once

#include <opencv2/videoio.hpp>
#include "Camera.hpp"

class RealCamera : public Camera {

    cv::VideoCapture cap;

public:
    RealCamera(int id);

    bool read(cv::OutputArray img);
};
