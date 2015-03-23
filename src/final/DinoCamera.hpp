#pragma once

#include "Camera.hpp"

class DinoCamera : public Camera {
    long frame = 0;
    std::string path;

public:
    int lastFrame = -1;
    int firstFrame = 1;

    void reset() {
        frame = firstFrame;
    }

    DinoCamera();

    bool read(cv::OutputArray img);

    virtual bool grab();

    virtual bool retrieve(cv::OutputArray img);
};
