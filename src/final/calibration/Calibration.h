#pragma once


#include "../camera/Camera.hpp"

class Calibration {

    cv::Size boardSize;
    float squareSize;

    cv::Ptr<Camera> camera;
    std::vector<std::vector<cv::Point2f>> everyCorners;
    cv::Mat lastFrame;

    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;

public:
    Calibration(cv::Ptr<Camera> camera) : camera(camera), boardSize(9, 6), squareSize(1.0f) {
    };

    bool acquireFrame();

    bool calibrate();

    void drawChessboardCorners(cv::Mat& image);

    void save(const std::string &filename);

};

