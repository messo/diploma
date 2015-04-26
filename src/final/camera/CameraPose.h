#pragma once

#include <opencv2/core/mat.hpp>

class CameraPose {

public:
    cv::Mat rvec, tvec;

    void load(const std::string& fileName);

    void save(const std::string& fileName) const;

    cv::Matx34d getRT() const;

    cv::Matx44d getPoseForPcl() const;

    void copyTo(CameraPose& to) const;
};
