#include "MOG2ForegroundMaskCalculator.h"

cv::Mat MOG2ForegroundMaskCalculator::calculate(cv::Mat nextFrame) {
    cv::Mat mask;
    bgSub->apply(nextFrame, mask, 0.001);

    removeShadows(mask);
    morph(mask);

    return mask;
}

void MOG2ForegroundMaskCalculator::removeShadows(cv::Mat mask) const {
    for (int y = 0; y < mask.rows; y++) {
        for (int x = 0; x < mask.cols; x++) {
            if (mask.at<uchar>(y, x) != 255) {
                mask.at<uchar>(y, x) = 0;
            }
        }
    }
}
