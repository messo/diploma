//
// Created by balint on 2015.03.22..
//

#include "FakeCamera.h"


FakeCamera::FakeCamera(int id) : Camera(id) {
    K = cv::Mat(cv::Matx33d(20, 0, 320,
            0, 20, 240,
            0, 0, 1));

    cv::invert(K, Kinv);

    distCoeffs = cv::Mat(cv::Matx14d(0, 0, 0, 0));
}

bool FakeCamera::read(cv::_OutputArray const &img) {
    return false;
}

bool FakeCamera::grab() {
    return false;
}

bool FakeCamera::retrieve(cv::_OutputArray const &img) {
    return false;
}
