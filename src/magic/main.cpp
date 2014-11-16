#include <iostream>

#include <opencv2/video/background_segm.hpp>
#include <GL/freeglut_std.h>

#define FPS_ENABLED

#include "Calibration.hpp"
#include "StereoCamera.hpp"
#include "OpenGLRenderer.hpp"

#define DEPTH_ENABLED true

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
    glutInit(&argc, argv);

    Ptr<Calibration> calibration(new Calibration("/home/balint/images/intrinsics.yml", "/home/balint/images/extrinsics.yml"));
    Ptr<StereoCamera> sc(new StereoCamera(StereoCamera::Type::DUMMY, calibration));

    Ptr<BackgroundSubtractorMOG2> bgSubLeft = createBackgroundSubtractorMOG2(500, 5.0);
    Ptr<BackgroundSubtractorMOG2> bgSubRight = createBackgroundSubtractorMOG2(500, 5.0);
    int erosion_size = 3;
    Mat kernel = getStructuringElement(MORPH_RECT,
            Size(2 * erosion_size + 1, 2 * erosion_size + 1),
            Point(erosion_size, erosion_size));

    OpenGLRenderer renderer(sc);
    renderer.init();

    while (true) {
        Mat left, right;
        left = sc->getLeft();
        right = sc->getRight();

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
            // DISPARITY
            Mat disp = sc->getDisparityMatrix(left, right);
            Mat maskedDisp;
            disp.copyTo(maskedDisp, leftMask);

            // 3D reconstruction
            if (sc->reprojectTo3D(maskedDisp)) {
                // OPENGL
                renderer.setProjection(calibration->cameraMatrix[0]);
                renderer.updatePoints(&sc->objectPoints, &sc->imagePoints, &left);
                renderer.render();
            }

            // visualization
            Mat normDisp = sc->normalizeDisparity(maskedDisp);
            imshow("maskedDisp", normDisp(sc->dispRoi));
        }

        //rectangle( leftMask, boundingRect.tl(), boundingRect.br(), Scalar( 255, 0, 0 ), 2, 8, 0 );
        //rectangle( rightMask, boundingRect.tl(), boundingRect.br(), Scalar( 255, 0, 0 ), 2, 8, 0 );

        //imshow("left", leftMask);
        //imshow("right", rightMask);
        imshow("left_color", left);
        //imshow("right_color", right);

        char ch = waitKey(5);
        if (ch != -1) {
            if (ch == 'w') {
                renderer.moveCamera(-5.0);
            } else if (ch == 's') {
                renderer.moveCamera(+5.0);
            }

            if (ch == 'q') {
                renderer.rotCameraX(-10.0);
            } else if (ch == 'e') {
                renderer.rotCameraX(+10.0);
            }

            if (ch == 'a') {
                renderer.rotCameraY(-10.0);
            } else if (ch == 'd') {
                renderer.rotCameraY(+10.0);
            }

            if(ch == 'i') {
                renderer.moveCenterY(1.0);
            } else if(ch == 'k') {
                renderer.moveCenterY(-1.0);
            }

            if(ch == 'j') {
                renderer.moveCenterX(-1.0);
            } else if(ch == 'l') {
                renderer.moveCenterX(1.0);
            }

            if(ch == 'r') {
                renderer.resetCamera();
            }

            if (ch == 27) {
                return 0;
            }
        }
    }
}
