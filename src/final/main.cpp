#include <iostream>

#include <opencv2/video/background_segm.hpp>
#include <opencv2/video/tracking.hpp>
#include <GL/freeglut_std.h>

#define FPS_ENABLED

#include "Calibration.hpp"
#include "StereoCamera.hpp"
#include "OpenGLRenderer.hpp"
#include "DummyCamera.hpp"
#include "ObjectSelector.hpp"

#define DEPTH_ENABLED true

using namespace std;
using namespace cv;

void clearImage(Mat mat, Rect rect) {
    rectangle(mat, rect.tl(), rect.br(), Scalar(0, 0, 0), -1, LINE_8, 0);
}

void dilateAndErode(Mat img, int erosion_size) {
    Mat kernel = getStructuringElement(MORPH_RECT,
            Size(2 * erosion_size + 1, 2 * erosion_size + 1),
            Point(erosion_size, erosion_size));

    dilate(img, img, kernel);
    erode(img, img, kernel);
}

void erodeAndDilate(Mat img, int erosion_size) {
    Mat kernel = getStructuringElement(MORPH_RECT,
            Size(2 * erosion_size + 1, 2 * erosion_size + 1),
            Point(erosion_size, erosion_size));

    erode(img, img, kernel);
    dilate(img, img, kernel);
}

void erode(Mat img, int erosion_size) {
    Mat kernel = getStructuringElement(MORPH_RECT,
            Size(2 * erosion_size + 1, 2 * erosion_size + 1),
            Point(erosion_size, erosion_size));

    erode(img, img, kernel);
}

Mat mergeImages(const Mat& left, const Mat& right) {
    Mat result = Mat::zeros(left.rows, left.cols + right.cols, left.type());

    left.copyTo(result(Rect(0, 0, left.cols, left.rows)));
    right.copyTo(result(Rect(left.cols - 1, 0, right.cols, right.rows)));

    return result;
}

int main(int argc, char **argv) {
    Ptr<DummyCamera> camera(new DummyCamera(Camera::LEFT, "/media/balint/Data/Linux/diploma/src/imgs_stereo_depth_single", 302));

    /// ----- MODELL ÉPÍTÉS!!!
    Ptr<BackgroundSubtractorMOG2> bgSub = createBackgroundSubtractorMOG2(302, 25.0, false);
    // dummy part, build a proper model, this cannot be used in real-time magic!
    for(int i=0; i<500; i++) {
        Mat current, mask;
        camera->retrieve(current);
        bgSub->apply(current, mask);
    }
    camera->firstFrame = 60;
    camera->reset();
    /// ------ MODELL KÉSZ

    // előző kép lekérdezése
    Mat prev, current, mask;
    camera->retrieve(prev);

    Ptr<ObjectSelector> objSelector(new ObjectSelector());

    //VideoWriter outputVideo;
    //outputVideo.open("/media/balint/Data/Linux/diploma/src/magic.mkv", VideoWriter::fourcc('X', 'V', 'I', 'D'), 15.0, Size(1280, 480), true);

    while (true) {
        // aktuális kép...
        camera->retrieve(current);
        bgSub->apply(current, mask, 0);

        // objektum kiválasztása...

        int niters = 3;
        dilate(mask, mask, Mat(), Point(-1,-1), niters);
        erode(mask, mask, Mat(), Point(-1,-1), niters*2);
        dilate(mask, mask, Mat(), Point(-1,-1), niters);

        //erodeAndDilate(mask, 2);
        //dilateAndErode(mask, 25);
        //erode(mask, 5);

        // --- CONNECTED COMPONENTS
        Mat selected(objSelector->selectUsingContoursWithClosestCentroid(current, mask));
        imshow("selected", selected);
        // ------------------------

        // --- CONTOURS
//        vector<vector<Point>> contours;
//        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
//        RNG rng(12345);
//
//        Mat dst;
//        current.copyTo(dst);
//        for (int i = 0; i < contours.size(); i++) {
//            Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
//            drawContours(dst, contours, i, color, 2, 8);
//        }
//        imshow("mask", dst);
        // ------------------------

        imshow("current", current);

        Mat merged(mergeImages(current, selected));
        imshow("merged", merged);

        //outputVideo << merged;

        Mat prevGray, currentGray, flow;
        cvtColor(prev, prevGray, cv::COLOR_BGR2GRAY);
        cvtColor(current, currentGray, cv::COLOR_BGR2GRAY);
        //calcOpticalFlowFarneback(prevGray, currentGray, flow, 0.5, 2, 13, 10, 5, 1.1, 0);

        current.copyTo(prev);

        char key = waitKey(50);
        if (key == 27) {
            //outputVideo.release();
            return 0;
        }
    }
}
