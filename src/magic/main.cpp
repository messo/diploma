#include <iostream>

#include <opencv2/video/background_segm.hpp>

#define FPS_ENABLED

#include "Calibration.hpp"
#include "StereoCamera.hpp"

#define DEPTH_ENABLED true

using namespace std;
using namespace cv;

int main() {
    cv::Ptr<Calibration> calibration(new Calibration("/home/balint/images/intrinsics.yml", "/home/balint/images/extrinsics.yml"));
    StereoCamera sc(StereoCamera::Type::REAL, calibration);

    Ptr<BackgroundSubtractorMOG2> bgSubLeft = createBackgroundSubtractorMOG2(500, 5.0);
    Ptr<BackgroundSubtractorMOG2> bgSubRight = createBackgroundSubtractorMOG2(500, 5.0);
    int erosion_size = 3;
    Mat kernel = getStructuringElement(MORPH_RECT,
            Size(2 * erosion_size + 1, 2 * erosion_size + 1),
            Point(erosion_size, erosion_size));

    while (true) {
        Mat left, right;
        left = sc.getLeft();
        right = sc.getRight();

        Mat leftMask, rightMask;
        bgSubLeft->apply(left, leftMask);
        erode(leftMask, leftMask, kernel);
        dilate(leftMask, leftMask, kernel);
        bgSubRight->apply(right, rightMask);
        erode(rightMask, rightMask, kernel);
        dilate(rightMask, rightMask, kernel);

        Rect bRectLeft = boundingRect(leftMask);
        Rect bRectRight = boundingRect(rightMask);
        Rect boundingRect(bRectLeft.tl(), bRectRight.br());

        if (DEPTH_ENABLED) {
            Mat disp = sc.getDisparityMatrix(left, right);
            Mat normDisp = sc.normalizeDisparity(disp);
            Mat maskedNormDisp;
            //normDisp.copyTo(maskedNormDisp, leftMask);
            imshow("maskedNormDisp", normDisp(sc.dispRoi));
        }

        //rectangle( leftMask, boundingRect.tl(), boundingRect.br(), Scalar( 255, 0, 0 ), 2, 8, 0 );
        //rectangle( rightMask, boundingRect.tl(), boundingRect.br(), Scalar( 255, 0, 0 ), 2, 8, 0 );

        //imshow("left", leftMask);
        //imshow("right", rightMask);
        imshow("left_color", left);
        imshow("right_color", right);

        char ch = waitKey(5);
        if (ch != -1) {
            if (ch == 27) {
                return 0;
            }
        }
    }
}
