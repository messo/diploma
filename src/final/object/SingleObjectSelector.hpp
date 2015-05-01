#pragma once

#include "ObjectSelector.h"
#include "../camera/Camera.hpp"
#include "Matcher.h"

class SingleObjectSelector : public ObjectSelector {

public:

    SingleObjectSelector(const Matcher &matcher) : ObjectSelector(matcher) { }

    virtual std::vector<Object> selectObjects(const std::vector<cv::Mat> &frames,
                                              const std::vector<cv::Mat> &masks) override;

};
