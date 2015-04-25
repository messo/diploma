#include <iostream>

#include "SURFFeatureExtractor.h"

using namespace cv;
using namespace xfeatures2d;
using namespace std;

SURFFeatureExtractor::SURFFeatureExtractor(const vector<cv::Mat> &images, const vector<cv::Mat> &masks) {

    extractor = Ptr<SURF>(new SURF());

    keypoints.resize(images.size());
    descriptors.resize(images.size());

    int minHessian = 400;
    SurfFeatureDetector detector(minHessian);

    for (int i = 0; i < images.size(); i++) {
//        GpuMat _m;
//        _m.upload(images[i]);

        detector.detect(images[i], keypoints[i], masks[i]);

        extractor->compute(images[i], keypoints[i], descriptors[i]);

//        (*extractor)(_m, GpuMat(), imgpts[i], descriptors[i], false);

    }
}
