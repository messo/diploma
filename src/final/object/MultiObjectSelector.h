#pragma once

#include "ObjectSelector.h"
#include "../camera/Camera.hpp"
#include "Blob.h"
#include "Object.h"
#include "Matcher.h"

class MultiObjectSelector : public ObjectSelector {

    const double MIN_AREA = 400.0;

    std::vector<std::vector<cv::Point>> getContours(const cv::Mat &mask);

    cv::Mat getTotalMask(cv::Size size, std::vector<std::vector<cv::Point>> &vector);

public:

    MultiObjectSelector(Matcher &matcher) : ObjectSelector(matcher) { }

    virtual std::vector<Object> selectObjects(const std::vector<cv::Mat> &frames,
                                              const std::vector<cv::Mat> &masks) override;

};
