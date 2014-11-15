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
};
