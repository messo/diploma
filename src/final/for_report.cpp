#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "camera/Camera.hpp"
#include "camera/RealCamera.hpp"
#include "calibration/CameraPoseCalculator.h"
#include "Common.h"
#include "optical_flow/ReportOpticalFlowCalculator.h"

using namespace cv;
using namespace std;

int main_mask(int argc, char **argv) {

    Mat img = imread("/media/balint/Data/cucc/mask343.png");
    Mat mask;
    cvtColor(img, mask, COLOR_RGB2GRAY);

    Mat maskWithoutShadows = mask.clone();
    for (int y = 0; y < maskWithoutShadows.rows; y++) {
        for (int x = 0; x < maskWithoutShadows.cols; x++) {
            if (maskWithoutShadows.at<uchar>(y, x) != 255) {
                maskWithoutShadows.at<uchar>(y, x) = 0;
            }
        }
    }

    Mat result = maskWithoutShadows.clone();

    int niters = 3;

    Mat small = getStructuringElement(MORPH_RECT, Size(2, 2));
    Mat bigger = getStructuringElement(MORPH_RECT, Size(5, 5));

    erode(result, result, small, Point(-1, -1), niters);
    dilate(result, result, bigger, Point(-1, -1), niters * 2);
    erode(result, result, bigger, Point(-1, -1), niters * 2);

//    erode(result, result, Mat(), Point(-1, -1), niters * 2);
//    dilate(result, result, Mat(), Point(-1, -1), niters);
//    erode(result, result, Mat(), Point(-1, -1), niters);

    imwrite("/media/balint/Data/cucc/mask343_fixed.png", result);

    Mat original = imread("/media/balint/Data/cucc/image343.png");
    Mat applied;
    original.copyTo(applied, result);
    imwrite("/media/balint/Data/cucc/mask343_applied.png", applied);


    vector<vector<Point>> contours;

    findContours(result, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat contourResult(480, 640, CV_8U, Scalar(0));

    if (contours.size() != 0) {
        // iterate through all the top-level contours,
        // draw each connected component with its own random color
        int largestComp = 0;
        double maxArea = 0;

        for (int idx = 0; idx < contours.size(); idx++) {
            const vector<Point> &c = contours[idx];
            double area = fabs(contourArea(Mat(c)));
            if (area > maxArea) {
                maxArea = area;
                largestComp = idx;
            }
        }

        for (int idx = 0; idx < contours.size(); idx++) {
            if (idx == largestComp) {
                drawContours(contourResult, contours, idx, Scalar(255), FILLED, LINE_8);
            } else {
                drawContours(contourResult, contours, idx, Scalar(255), 1, LINE_AA);
            }
        }
    }

    imwrite("/media/balint/Data/cucc/mask343_contours.png", contourResult);

    while (true) {
        imshow("original", mask);
        imshow("maskWithoutShadows", maskWithoutShadows);
        imshow("result", result);
        imshow("contourResult", contourResult);

        char ch = waitKey(50);
        if (ch == 27) {
            break;
        }
    }

    return 0;
}


int main_pose(int argc, char **argv) {

    int camId = Camera::RIGHT;
    Ptr<Camera> camera(new RealCamera(camId, "/media/balint/Data/Linux/diploma/src/final/intrinsics_right.yml"));
    CameraPoseCalculator calculator(camera);

    int count = 0;
    while (true) {
        Mat image;
        camera->readUndistorted(image);
        if (calculator.poseCalculated()) {
            drawGridXY(image, camera, calculator.cameraPose);
            // drawBoxOnChessboard(image, camera, calculator.cameraPose);
        }
        imshow("image", image);

        char ch = (char) waitKey(33);
        count++;

        if (ch == 27) {
            break;
        } else if (ch == 'p') {
            if (calculator.calculate()) {
                std::cout << "Calculated." << std::endl;
            } else {
                std::cout << "Not calculated!" << std::endl;
            }
        } else if (calculator.poseCalculated() && (count % 90 == 0)) {
            imwrite("/media/balint/Data/Linux/diploma/pose" + to_string(camId) + "_" + to_string(count) + ".png",
                    image);
        }
    }

    return 0;
}

int main(int argc, char **argv) {

    vector<Ptr<Camera>> camera(2);
    camera[Camera::LEFT] = Ptr<Camera>(
            new RealCamera(Camera::LEFT, "/media/balint/Data/Linux/diploma/src/final/intrinsics_left.yml"));
    camera[Camera::RIGHT] = Ptr<Camera>(
            new RealCamera(Camera::RIGHT, "/media/balint/Data/Linux/diploma/src/final/intrinsics_right.yml"));

    vector<CameraPose> cameraPose(2);
    cameraPose[Camera::LEFT].load("/media/balint/Data/Linux/diploma/src/final/pose_left.yml");
    cameraPose[Camera::RIGHT].load("/media/balint/Data/Linux/diploma/src/final/pose_right.yml");

    std::vector<Ptr<BackgroundSubtractorMOG2>> bgSub(2);
    bgSub[Camera::LEFT] = createBackgroundSubtractorMOG2(300, 25.0, true);
    bgSub[Camera::RIGHT] = createBackgroundSubtractorMOG2(300, 25.0, true);

    int focus = 80;
    static_cast<RealCamera *>(camera[Camera::LEFT].get())->focus(focus);
    static_cast<RealCamera *>(camera[Camera::RIGHT].get())->focus(focus);

    std::vector<SingleObjectSelector> objSelector(2);


    ReportOpticalFlowCalculator ofCalculator(camera[Camera::LEFT], camera[Camera::RIGHT]);

    Mat left = imread("/media/balint/Data/Linux/diploma/of_img_left.png");
    Mat right = imread("/media/balint/Data/Linux/diploma/of_img_right.png");
    Mat maskLeft = imread("/media/balint/Data/Linux/diploma/of_mask_left.png", IMREAD_GRAYSCALE);
    Mat maskRight = imread("/media/balint/Data/Linux/diploma/of_mask_right.png", IMREAD_GRAYSCALE);

    vector<Mat> frames(2);
    frames[0] = left;
    frames[1] = right;

    vector<Mat> masks(2);
    masks[0] = maskLeft;
    masks[1] = maskRight;

    ofCalculator.feed(frames, masks);

    char ch = (char) waitKey();
    return 0;
}