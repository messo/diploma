#pragma once

#include <map>
#include <opencv2/core/matx.hpp>
#include "Camera.hpp"
#include "Common.h"

class MultiView {

    cv::Ptr<Camera> cam;

    std::map<int, cv::Matx34d> Pmats;

public:

    Cloud cloud;

    MultiView(cv::Ptr<Camera> camera);

    void addP(long frameId, cv::Matx34d P);

    void reconstructNext(long frameId1, const std::vector<cv::Point2f> &points1, long frameId2,
                         const std::vector<cv::Point2f> &points2);
};
