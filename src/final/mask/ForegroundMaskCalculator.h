#pragma once

#include <opencv2/imgproc.hpp>

class ForegroundMaskCalculator {

public:
    virtual cv::Mat calculate(cv::Mat nextFrame) = 0;

    void morph(cv::Mat &mask) {
        int niters = 3;
        dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), niters);
        erode(mask, mask, cv::Mat(), cv::Point(-1, -1), niters * 2);
        dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), niters);
    }
};
