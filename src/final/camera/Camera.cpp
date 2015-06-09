#include <iostream>
#include <opencv2/imgproc.hpp>
#include "Camera.hpp"

void Camera::readCalibration(const std::string &calibrationFile) {
    cv::FileStorage fs;
    if (!fs.open(calibrationFile, cv::FileStorage::READ)) {
        throw "File cannot be opened!";
    }

    if (id == 0) {
        fs["M1"] >> cameraMatrix;
        fs["D1"] >> distCoeffs;
    } else if(id == 1) {
        fs["M2"] >> cameraMatrix;
        fs["D2"] >> distCoeffs;
    }

    invert(cameraMatrix, Kinv);
}

bool Camera::readUndistorted(cv::OutputArray img) {
    cv::Mat distorted;
    read(distorted);
    undistort(distorted, img, cameraMatrix, distCoeffs);
    return true;
}
