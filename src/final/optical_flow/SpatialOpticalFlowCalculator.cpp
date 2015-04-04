#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui.hpp>
#include "../Common.h"
#include "SpatialOpticalFlowCalculator.h"

using namespace cv;
using namespace std;


bool SpatialOpticalFlowCalculator::feed(const Mat frame1, const Mat mask1,
                                        const Mat frame2, const Mat mask2) {

    if (frame1.channels() != 1) {
        cvtColor(frame1, this->frame1, COLOR_BGR2GRAY);
    } else {
        this->frame1 = frame1;
    }
    this->mask1 = mask1;
    this->calcTexturedRegions(this->frame1, mask1, texturedRegions1);
    //imshow("texturedRegions1", texturedRegions1);

    if (frame2.channels() != 1) {
        cvtColor(frame2, this->frame2, COLOR_BGR2GRAY);
    } else {
        this->frame1 = frame2;
    }
    this->mask2 = frame2;
    this->calcTexturedRegions(this->frame2, mask2, texturedRegions2);
    //imshow("texturedRegions2", texturedRegions2);

    this->calcOpticalFlow();
    this->visualizeMatches(points1, points2);

    return true;
}
