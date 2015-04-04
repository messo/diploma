#include <iostream>
#include <opencv2/imgproc.hpp>
#include "Camera.hpp"

void Camera::readCalibration(const std::string &calibrationFile) {
    cv::FileStorage fs;
    if (!fs.open(calibrationFile, cv::FileStorage::READ)) {
        throw "File cannot be opened!";
    }

    if (id == LEFT) {
        fs["M1"] >> K;
        fs["D1"] >> distCoeffs;
    } else {
        fs["M2"] >> K;
        fs["D2"] >> distCoeffs;
    }

    invert(K, Kinv);
}

bool Camera::readUndistorted(cv::OutputArray img) {
    cv::Mat distorted;
    read(distorted);
    undistort(distorted, img, K, distCoeffs);
    return true;
}
