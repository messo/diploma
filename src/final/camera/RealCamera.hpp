#pragma once

#include <opencv2/videoio.hpp>
#include "Camera.hpp"

class RealCamera : public Camera {

    cv::VideoCapture cap;

public:
    RealCamera(int id);

    RealCamera(int id, const std::string &calibrationFile);

    bool read(cv::OutputArray img);

    virtual bool grab();

    virtual bool retrieve(cv::OutputArray img);
};
