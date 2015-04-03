#pragma once

#include <map>
#include <opencv2/core/matx.hpp>
#include "camera/Camera.hpp"
#include "Common.h"

class MultiView {

    cv::Ptr<Camera> cam;

    std::map<int, cv::Matx34d> Pmats;

    int frame = 1;

public:

    Cloud cloud;

    MultiView(cv::Ptr<Camera> camera);

    cv::Matx34d P(long frameId);

    void addP(long frameId, cv::Matx34d P);

    void reconstructNext(long frameId1, const std::vector<cv::Point2f> &points1, long frameId2,
                         const std::vector<cv::Point2f> &points2);
};
