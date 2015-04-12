#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui.hpp>
#include "../Common.h"
#include "SpatialOpticalFlowCalculator.h"

using namespace cv;
using namespace std;


bool SpatialOpticalFlowCalculator::feed(cv::Mat(&frames)[2], ObjectSelector(&objSelector)[2]) {

    Point translations[2];

    double t0 = getTickCount();

#pragma omp parallel for
    for (int i = 0; i < 2; i++) {
        if (frames[i].channels() != 1) {
            cvtColor(frames[i], this->frames[i], COLOR_BGR2GRAY);
        } else {
            this->frames[i] = frames[i].clone();
        }
        this->masks[i] = objSelector[i].lastMask.clone();

        // move the frame2 and mask around a bit, so the main object's movement is not that big -- so OF will be okay
        translations[i] = moveToTheCenter(this->frames[i], this->masks[i]);

        this->calcTexturedRegions(this->frames[i], this->masks[i], this->texturedRegions[i]);
    }

    t0 = ((double) getTickCount() - t0) / getTickFrequency();
    std::cout << "Feed init done in " << t0 << "s" << std::endl;
    std::cout.flush();

    this->calcOpticalFlow(translations[1]);

//    this->visualizeMatches(points1, points2);

    // move the points to their original location
    for (int i = 0; i < points1.size(); i++) {
        points1[i] -= Point2f(translations[0]);
        points2[i] -= Point2f(translations[1]);
    }

//    this->visualizeMatches(frames[0], points1, frames[1], points2);

    return true;
}
