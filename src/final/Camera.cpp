#include <iostream>
#include "Camera.hpp"

using namespace cv;
using namespace std;

void Camera::readCalibration(string calibrationFile) {
    FileStorage fs;
    fs.open(calibrationFile, FileStorage::READ);

    if(id == LEFT) {
        fs["M1"] >> K;
        fs["D1"] >> distCoeff;
    } else {
        fs["M2"] >> K;
        fs["D2"] >> distCoeff;
    }

    invert(K, Kinv);
}
