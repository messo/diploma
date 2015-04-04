#include <opencv2/video/tracking.hpp>
#include "TemporalOpticalFlowCalculator.h"

using namespace cv;
using namespace std;


bool TemporalOpticalFlowCalculator::feed(long frameId, const Mat &image, const Mat &lastMask, const Rect &boundingRect,
                                         const vector<Point> &contour) {

    // --------------------------------------
    // 1. translate the image to the "center"
    // --------------------------------------

    Point translation((320 - boundingRect.width / 2) - boundingRect.x,
                      (240 - boundingRect.height / 2) - boundingRect.y);

    currentBoundingRect = boundingRect + translation;

    // translate the image
    shiftImage(image, boundingRect, translation, frame2);
    // translate the mask
    shiftImage(lastMask, boundingRect, translation, mask2);
    // translating the contour
    currentContour.clear();
    translate(contour, translation, currentContour);

    // calculating textured regions
    calcTexturedRegions(frame2, mask2, texturedRegions2);

    Point t;
    if (frame1.empty() || (calcOpticalFlow(t)) > 1.0) {
        // -----------------------------
        // 2.
        // -----------------------------

        // shiftelés csak akkor, ha elfogadjuk, egyébként következő képkocka ("túl kicsi mozgás")

        bool prevFrameEmpty = frame1.empty();

        prevFrameId = currentFrameId;
        currentFrameId = frameId;

        prevContour = currentContour;
        frame2.copyTo(frame1);
        mask2.copyTo(mask1);
        texturedRegions2.copyTo(texturedRegions1);

        return !prevFrameEmpty;
    } else {
        return false;
    }
}
