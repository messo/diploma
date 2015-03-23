#pragma once

#include <opencv2/videoio.hpp>

class Camera {

    int id;

public:

    cv::Mat K;
    cv::Mat Kinv;
    cv::Mat distCoeff;

    static const int LEFT = 0;
    static const int RIGHT = 1;

    Camera(int id) : id(id) {
    }

    int getId() const {
        return id;
    }

    void readCalibration(std::string calibrationFile);

    virtual bool read(cv::OutputArray img) = 0;

    virtual bool grab() = 0;

    virtual bool retrieve(cv::OutputArray img) = 0;
};
