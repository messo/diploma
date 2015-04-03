#pragma once

#include <iostream>
#include <opencv2/videoio.hpp>

class Camera {

    int id;

public:

    cv::Mat K;
    cv::Mat Kinv;
    cv::Mat distCoeffs;

    static const int LEFT = 0;
    static const int RIGHT = 1;

    Camera(int id) : id(id) {
    }

    int getId() const {
        return id;
    }

    void readCalibration(std::string calibrationFile);

    double getWidth() { return 640.0; }

    double getHeight() { return 480.0; }

    double getFocalLength() { return K.at<double>(0, 0); }

    virtual bool read(cv::OutputArray img) = 0;

    bool readUndistorted(cv::OutputArray img);

    virtual bool grab() = 0;

    virtual bool retrieve(cv::OutputArray img) = 0;

};
