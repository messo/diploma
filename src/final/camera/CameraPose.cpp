#include <opencv2/core/persistence.hpp>
#include "CameraPose.h"

void CameraPose::load(const std::string &fileName) {
    cv::FileStorage fs;
    fs.open(fileName, cv::FileStorage::READ);

    fs["rvec"] >> rvec;
    fs["tvec"] >> tvec;
}

void CameraPose::save(const std::string &fileName) {
    cv::FileStorage fs;
    fs.open(fileName, cv::FileStorage::WRITE);

    fs << "rvec" << rvec;
    fs << "tvec" << tvec;
}
