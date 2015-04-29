#pragma once

#include "ObjectSelector.h"
#include "../camera/Camera.hpp"
#include "Blob.h"
#include "Object.h"
#include "Matcher.h"

class MultiObjectSelector : public ObjectSelector {

    const double MIN_AREA = 400.0;

    Matcher matcher;

    std::vector<std::vector<Blob>> blobs;
    std::vector<std::vector<std::vector<cv::Point>>> contours;

    std::vector<std::vector<cv::Point>> getContours(const cv::Mat &mask);

    cv::Mat getTotalMask(std::vector<std::vector<cv::Point>> &vector);

public:

    MultiObjectSelector(const cv::Ptr<Camera> &camera1, const cv::Ptr<Camera> &camera2, const cv::Mat &F) :
            contours(2), blobs(2), matcher(camera1, camera2, F) {
    }

    virtual std::vector<Object> selectObjects(const std::vector<cv::Mat> &frames,
                                              const std::vector<cv::Mat> &masks) override;
};
