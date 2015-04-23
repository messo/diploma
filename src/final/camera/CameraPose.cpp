#include <opencv2/core/persistence.hpp>
#include <opencv2/calib3d.hpp>
#include "CameraPose.h"

void CameraPose::load(const std::string &fileName) {
    cv::FileStorage fs;
    fs.open(fileName, cv::FileStorage::READ);

    fs["rvec"] >> rvec;
    fs["tvec"] >> tvec;
}

void CameraPose::save(const std::string &fileName) const {
    cv::FileStorage fs;
    fs.open(fileName, cv::FileStorage::WRITE);

    fs << "rvec" << rvec;
    fs << "tvec" << tvec;
}

cv::Matx34d CameraPose::getRT() const {
    cv::Matx33d R;
    cv::Rodrigues(this->rvec, R);
    return cv::Matx34d(R(0, 0), R(0, 1), R(0, 2), this->tvec.at<double>(0, 0),
                       R(1, 0), R(1, 1), R(1, 2), this->tvec.at<double>(1, 0),
                       R(2, 0), R(2, 1), R(2, 2), this->tvec.at<double>(2, 0));
}

cv::Matx44d CameraPose::getPoseForPcl() const {
    cv::Affine3d affine(this->rvec, this->tvec);

    cv::Matx33d magic(1.0, 0, 0,
                      0, 1.0, 0,
                      0, 0, -1.0);

    return affine.rotate(magic).matrix;
}
