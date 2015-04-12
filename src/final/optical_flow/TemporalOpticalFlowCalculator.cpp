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
    shiftImage(image, boundingRect, translation, frames[1]);
    // translate the mask
    shiftImage(lastMask, boundingRect, translation, masks[1]);
    // translating the contour
    currentContour.clear();
    translate(contour, translation, currentContour);

    // calculating textured regions
    calcTexturedRegions(frames[1], masks[1], texturedRegions[1]);

    Point t;
    if (frames[0].empty() || (calcOpticalFlow(t)) > 1.0) {
        // -----------------------------
        // 2.
        // -----------------------------

        // shiftelés csak akkor, ha elfogadjuk, egyébként következő képkocka ("túl kicsi mozgás")

        bool prevFrameEmpty = frames[0].empty();

        prevFrameId = currentFrameId;
        currentFrameId = frameId;

        prevContour = currentContour;
        frames[1].copyTo(frames[0]);
        masks[1].copyTo(masks[0]);
        texturedRegions[1].copyTo(texturedRegions[0]);

        return !prevFrameEmpty;
    } else {
        return false;
    }
}
