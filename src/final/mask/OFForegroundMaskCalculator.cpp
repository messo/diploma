#include "OFForegroundMaskCalculator.h"

#include <opencv2/video/tracking.hpp>
#include <iostream>
#include <iomanip>
#include <opencv2/cudaoptflow.hpp>

cv::Mat OFForegroundMaskCalculator::calculate(cv::Mat nextFrame_) {
    double t0 = cv::getTickCount();

    cv::Mat nextFrame;
    if (nextFrame_.channels() != 1) {
        cv::cvtColor(nextFrame_, nextFrame, cv::COLOR_BGR2GRAY);
    } else {
        nextFrame = nextFrame_;
    }

    if (previousFrame.empty()) {
        nextFrame.copyTo(previousFrame);
        return cv::Mat::zeros(nextFrame.rows, nextFrame.cols, CV_8U);
    }

    cv::cuda::GpuMat d_frameL(previousFrame), d_frameR(nextFrame);
    cv::cuda::GpuMat d_flow;
    cv::Ptr<cv::cuda::FarnebackOpticalFlow> of = cv::cuda::FarnebackOpticalFlow::create(1, 0.99, false, 3, 1, 5, 1.1, 0);
    //cv::calcOpticalFlowFarneback(previousFrame, nextFrame, flow, 0.99, 1, 3, 1, 3, 1.1, 0);

    of->calc(d_frameL, d_frameR, d_flow, cv::cuda::Stream::Null());

    cv::Mat flow;
    d_flow.download(flow);

    cv::Mat mask(this->getMaskFromFlow(flow));

    nextFrame.copyTo(previousFrame);

    t0 = ((double) cv::getTickCount() - t0) / cv::getTickFrequency();
    std::cout << "[" << std::setw(20) << "MaskCalculator" << "] " << "Mask calculated in: " << t0 << "s" << std::endl;

    return mask;
}

cv::Mat OFForegroundMaskCalculator::getMaskFromFlow(cv::Mat flow) {
    cv::Mat mask(flow.rows, flow.cols, CV_8U, cv::Scalar(0));

    cv::Rect valid(cv::Point(0, 0), flow.size());

    for (int i = 0; i < flow.rows; i++) {
        for (int j = 0; j < flow.cols; j++) {
            const cv::Point2f &v = flow.at<cv::Point2f>(i, j);

            if (cv::norm(v) > 1.0) {
                cv::Point coords(cv::Point(j, i) + cv::Point(v));
                if (coords.inside(valid)) {
                    mask.at<uchar>(coords) = 255;
                }
            }
        }
    }

    morph(mask);

    return mask;
}
