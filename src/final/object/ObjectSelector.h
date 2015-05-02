#pragma once

#include <opencv2/core.hpp>
#include "Object.h"
#include "Matcher.h"

class ObjectSelector {

protected:
    Matcher &matcher;

public:

    ObjectSelector(Matcher &matcher) : matcher(matcher) { }

    virtual std::vector<Object> selectObjects(const std::vector<cv::Mat> &frames,
                                              const std::vector<cv::Mat> &masks) = 0;
};
