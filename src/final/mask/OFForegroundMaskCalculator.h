#pragma once

#include "ForegroundMaskCalculator.h"

class OFForegroundMaskCalculator : public ForegroundMaskCalculator {

    cv::Mat previousFrame;

    cv::Mat getMaskFromFlow(cv::Mat mat);

public:
    virtual cv::Mat calculate(cv::Mat nextFrame);

};
