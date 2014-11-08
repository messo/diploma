#include <iostream>

#include <opencv2/video/background_segm.hpp>

#include "Calibration.hpp"
#include "RealStereoCamera.hpp"
#include "DummyStereoCamera.hpp"

using namespace std;

int main() {
    Calibration calibration("/home/balint/images/intrinsics.yml", "/home/balint/images/extrinsics.yml");
    DummyStereoCamera rsc(&calibration);

    Mat left, right;

    // 0. create the stuff
    Ptr<BackgroundSubtractorMOG2> bgSubLeft = createBackgroundSubtractorMOG2(500, 5.0);
    //Ptr<BackgroundSubtractorMOG2> bgSubRight = createBackgroundSubtractorMOG2(1);

    int erosion_size = 3;

    Mat kernel = getStructuringElement(MORPH_RECT,
            Size(2 * erosion_size + 1, 2 * erosion_size + 1),
            Point(erosion_size, erosion_size));

    // 2. apply the mask.
    while (true) {
        left = rsc.getLeft();
        right = rsc.getRight();

        Mat leftMask;
        bgSubLeft->apply(left, leftMask);
        erode(leftMask, leftMask, kernel);
        dilate(leftMask, leftMask, kernel);

        Mat disp = rsc.getDisparityMatrix(left, right);
        Mat normDisp = rsc.normalizeDisparity(disp);
        Mat maskedNormDisp;
        normDisp.copyTo(maskedNormDisp, leftMask);
        imshow("maskedNormDisp", maskedNormDisp);

        imshow("left", leftMask);

        imshow("left_color", left);

        char ch = waitKey(50);
        if (ch != -1) {
            if (ch == 27) {
                return 0;
            }
        }
    }
}
