#pragma once

#include "ObjectSelector.h"
#include "../camera/Camera.hpp"
#include "Matcher.h"

class SingleObjectSelector : public ObjectSelector {

    Matcher matcher;

    cv::Mat selectUsingConnectedComponents(const cv::Mat &img, const cv::Mat &mask);

    cv::Mat selectUsingContourWithMaxArea(const cv::Mat &img, cv::Mat mask);

public:

    SingleObjectSelector(const cv::Ptr<Camera> &camera1, const cv::Ptr<Camera> &camera2, const cv::Mat &F) :
            matcher(camera1, camera2, F) { }

    virtual std::vector<Object> selectObjects(const std::vector<cv::Mat> &frames,
                                              const std::vector<cv::Mat> &masks) override;

    cv::Mat lastMask;
};
