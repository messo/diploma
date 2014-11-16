#pragma once

#include <opencv2/videoio.hpp>

class Camera {

    int id;

public:
    Camera(int id) : id(id) {
    }

    int getId() const {
        return id;
    }

    virtual bool read(cv::OutputArray img) = 0;

    virtual bool grab() = 0;

    virtual bool retrieve(cv::OutputArray img) = 0;
};
