#pragma once

#include <opencv2/core.hpp>
#include "Object.h"

class ObjectSelector {

public:

    virtual std::vector<Object> selectObjects(const std::vector<cv::Mat> &frames,
                                              const std::vector<cv::Mat> &masks) = 0;
};
