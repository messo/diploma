#pragma once

#include "Camera.hpp"

class DummyCamera : public Camera {
    long frame = 0;

public:
    DummyCamera(int id);

    bool read(cv::OutputArray img);

    virtual bool grab();

    virtual bool retrieve(cv::OutputArray img);
};
