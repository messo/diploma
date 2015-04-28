#pragma once

#include <opencv2/core.hpp>
#include "../camera/Camera.hpp"
#include "Blob.h"
#include "Object.h"

class MultiObjectSelector {

    const double MIN_AREA = 300.0;

    const cv::Ptr<Camera> &camera1;
    const cv::Ptr<Camera> &camera2;
    const cv::Mat &F;

    std::vector<std::vector<Blob>> blobs;
    std::vector<std::vector<std::vector<cv::Point>>> contours;

    std::vector<std::vector<cv::Point>> getContours(const cv::Mat &mask);

    cv::Mat getTotalMask(std::vector<std::vector<cv::Point>> &vector);

public:

    std::vector<Object> objects;

    MultiObjectSelector(const cv::Ptr<Camera> &camera1, const cv::Ptr<Camera> &camera2, const cv::Mat &F) :
            camera1(camera1), camera2(camera2), F(F), contours(2), blobs(2) { }

    void selectObjects(std::vector<cv::Mat> masks, std::vector<cv::Mat> vector1);

};
