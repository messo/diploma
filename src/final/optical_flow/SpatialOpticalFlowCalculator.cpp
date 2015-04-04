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
        this->frame1 = frame1.clone();
    }
    this->mask1 = mask1.clone();

    if (frame2.channels() != 1) {
        cvtColor(frame2, this->frame2, COLOR_BGR2GRAY);
    } else {
        this->frame2 = frame2.clone();
    }
    this->mask2 = mask2.clone();

    // move the frame2 and mask around a bit, so the main object's movement is not that big -- so OF will be okay
    Point translation1 = moveToTheCenter(this->frame1, this->mask1);
    Point translation2 = moveToTheCenter(this->frame2, this->mask2);

    this->calcTexturedRegions(this->frame1, this->mask1, texturedRegions1);
    this->calcTexturedRegions(this->frame2, this->mask2, texturedRegions2);

    this->calcOpticalFlow(translation2);

    this->visualizeMatches(points1, points2);

    // move the points to their original location
    for(int i=0; i<points1.size(); i++) {
        points1[i] -= Point2f(translation1);
        points2[i] -= Point2f(translation2);
    }

    this->visualizeMatches(frame1, points1, frame2, points2);

    return true;
}
