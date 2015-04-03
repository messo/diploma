#include <iostream>
#include <opencv2/imgproc.hpp>
#include "Camera.hpp"

using namespace cv;
using namespace std;

void Camera::readCalibration(string calibrationFile) {
    FileStorage fs;
    if(!fs.open(calibrationFile, FileStorage::READ)) {
        throw "File cannot be opened!";
    }

    if(id == LEFT) {
        fs["M1"] >> K;
        fs["D1"] >> distCoeffs;
    } else {
        fs["M2"] >> K;
        fs["D2"] >> distCoeffs;
    }

    invert(K, Kinv);
}

bool Camera::readUndistorted(OutputArray img) {
    Mat distorted;
    read(distorted);
    undistort(distorted, img, K, distCoeffs);
    return true;
}
