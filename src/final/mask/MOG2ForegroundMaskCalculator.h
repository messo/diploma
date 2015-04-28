#pragma once

#include "ForegroundMaskCalculator.h"
#include <opencv2/video/background_segm.hpp>

class MOG2ForegroundMaskCalculator : public ForegroundMaskCalculator {

    cv::Ptr<cv::BackgroundSubtractorMOG2> bgSub;

    void removeShadows(cv::Mat mat) const;

public:

    MOG2ForegroundMaskCalculator() {
        bgSub = cv::createBackgroundSubtractorMOG2(300, 25.0, true);
    }

    virtual cv::Mat calculate(cv::Mat nextFrame);

};
