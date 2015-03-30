#pragma once

#include "Camera.hpp"

class DummyCamera : public Camera {
    long frame = 0;
    long currentFrame = 0;
    std::string path;

public:
    int lastFrame = -1;
    int firstFrame = 0;

    void reset() {
        frame = firstFrame;
        currentFrame = frame;
    }

    DummyCamera(int id, std::string path, int count);

    bool read(cv::OutputArray img);

    long getFrameId() const;

    virtual bool grab();

    virtual bool retrieve(cv::OutputArray img);
};
