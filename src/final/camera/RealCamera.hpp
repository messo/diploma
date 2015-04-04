#pragma once

#include <opencv2/videoio.hpp>
#include "Camera.hpp"

class RealCamera : public Camera {

    cv::VideoCapture cap;
    int descriptor;

public:
    RealCamera(int id);

    RealCamera(int id, const std::string &calibrationFile);

    void focus(int value);

    bool read(cv::OutputArray img);

    virtual bool grab();

    virtual bool retrieve(cv::OutputArray img);
};
