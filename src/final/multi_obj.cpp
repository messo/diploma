#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Geometry>

#include "Common.h"
#include "camera/Camera.hpp"
#include "camera/RealCamera.hpp"
#include "calibration/CameraPoseCalculator.h"
#include "mask/OFForegroundMaskCalculator.h"
#include "object/MultiObjectSelector.h"
#include "optical_flow/ReportOpticalFlowCalculator.h"
#include "Triangulator.h"
#include "Visualization.h"
#include "optical_flow/SpatialOpticalFlowCalculator.h"

using namespace cv;
using namespace std;


std::vector<std::vector<Mat>> getMasks(std::vector<Ptr<Camera>> &cameras,
                                       std::vector<Ptr<ForegroundMaskCalculator>> &maskCalculators) {

    std::vector<std::vector<Mat>> selected(2);

    selected[0].resize(2);
    selected[1].resize(2);

#pragma omp parallel for
    for (int i = 0; i < 2; i++) {
        // std::cout << "CAP THREAD: " << omp_get_thread_num() << std::endl;
        Mat image, mask;
        cameras[i]->read(image); // readUndistorted
        mask = maskCalculators[i]->calculate(image);

        Mat gray, equal;
        cvtColor(image, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, equal);

        selected[0][i] = equal;
        selected[1][i] = mask;
    }

    return selected;
}

enum VIS_TYPE {
    PIXELS, DEPTH, CONTOURS
};

int main(int argc, char **argv) {

    vector<Ptr<Camera>> camera(2);
    camera[Camera::LEFT] = Ptr<Camera>(
            new RealCamera(Camera::LEFT, "/media/balint/Data/Linux/diploma/src/final/intrinsics_left.yml"));
    camera[Camera::RIGHT] = Ptr<Camera>(
            new RealCamera(Camera::RIGHT, "/media/balint/Data/Linux/diploma/src/final/intrinsics_right.yml"));

    vector<CameraPose> cameraPose(2);
    cameraPose[Camera::LEFT].load("/media/balint/Data/Linux/diploma/src/final/pose_left.yml");
    cameraPose[Camera::RIGHT].load("/media/balint/Data/Linux/diploma/src/final/pose_right.yml");

    std::vector<Ptr<ForegroundMaskCalculator>> maskCalculators(2);
    maskCalculators[Camera::LEFT] = Ptr<ForegroundMaskCalculator>(new OFForegroundMaskCalculator());
    maskCalculators[Camera::RIGHT] = Ptr<ForegroundMaskCalculator>(new OFForegroundMaskCalculator());

    int focus = 80;
    static_cast<RealCamera *>(camera[Camera::LEFT].get())->focus(focus);
    static_cast<RealCamera *>(camera[Camera::RIGHT].get())->focus(focus);

    FileStorage fs;
    fs.open("/media/balint/Data/Linux/diploma/F.yml", FileStorage::READ);
    Mat F;
    fs["myF"] >> F;


    MultiObjectSelector objSelector(Matcher(camera[Camera::LEFT], camera[Camera::RIGHT], F));

//    long frame = 0;
//
//    while (true) {
//        //double t0 = getTickCount();
//
//        frame++;
//
//        std::vector<std::vector<Mat>> frames = getMasks(camera, maskCalculators);
//
//        imshow("frame1", frames[0][0]);
//        imshow("frame2", frames[0][1]);
//        imshow("mask1", frames[1][0]);
//        imshow("mask2", frames[1][1]);
//
//        char ch = (char) waitKey(33);
//        if (ch == 27) {
//            break;
//        } else if (frame % 20 == 0) {
//            imwrite("/media/balint/Data/Linux/multi_left.png", frames[0][0]);
//            imwrite("/media/balint/Data/Linux/multi_right.png", frames[0][1]);
//
//            imwrite("/media/balint/Data/Linux/multi_left_mask.png", frames[1][0]);
//            imwrite("/media/balint/Data/Linux/multi_right_mask.png", frames[1][1]);
//        }
//    }

    std::vector<Mat> frames(2);
    frames[0] = imread("/media/balint/Data/Linux/frame_left_230.png");//, IMREAD_GRAYSCALE);
    frames[1] = imread("/media/balint/Data/Linux/frame_right_230.png");//, IMREAD_GRAYSCALE);

    std::vector<Mat> masks(2);
    masks[0] = imread("/media/balint/Data/Linux/mask_left_230.png", IMREAD_GRAYSCALE);
    masks[1] = imread("/media/balint/Data/Linux/mask_right_230.png", IMREAD_GRAYSCALE);

//    std::vector<Mat> frames(2);
//    frames[0] = imread("/media/balint/Data/Linux/__MULTI/_multi_left.png", IMREAD_GRAYSCALE);
//    frames[1] = imread("/media/balint/Data/Linux/__MULTI/_multi_right.png", IMREAD_GRAYSCALE);
//
//    std::vector<Mat> masks(2);
//    masks[0] = imread("/media/balint/Data/Linux/__MULTI/_multi_left_mask.png", IMREAD_GRAYSCALE);
//    masks[1] = imread("/media/balint/Data/Linux/__MULTI/_multi_right_mask.png", IMREAD_GRAYSCALE);

    //while (true) {
    std::vector<Object> objects = objSelector.selectObjects(frames, masks);

    // tiny display.
    Mat leftResult(480, 640, CV_8UC3, Scalar(0, 0, 0));
    Mat rightResult(480, 640, CV_8UC3, Scalar(0, 0, 0));

    RNG rng;
    for (int i = 0; i < objects.size(); i++) {
        const Object &obj = objects[i];

        int icolor = (unsigned) rng;
        Scalar color(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
        Mat objMat(480, 640, CV_8UC3, color);

        objMat.copyTo(leftResult, obj.masks[0]);
        objMat.copyTo(rightResult, obj.masks[1]);
    }

    Mat merged = mergeImages(leftResult, rightResult);

    for (int i = 0; i < objects.size(); i++) {
        const Object &obj = objects[i];

        for (int j = 0; j < obj.matches.size(); j++) {
            int icolor = (unsigned) rng;
            Scalar color(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);

            circle(merged, obj.matches[j].first, 1, color, 1);
            circle(merged, obj.matches[j].second + Point2f(640, 0), 2, color, 1);
            line(merged, obj.matches[j].first, obj.matches[j].second + Point2f(640, 0), color, 1);
        }
    }

    imshow("objects", merged);

    SpatialOpticalFlowCalculator calculator(camera[0], camera[1], F);

    std::vector<CloudPoint> finalResult;
    std::vector<Point2f> totalPoints;

    for (int i = 0; i < objects.size(); i++) {
        imshow("OriginalFrame1", frames[0]);
        imshow("OriginalFrame2", frames[1]);

        calculator.feed(frames, objects[i]);

        Triangulator triangulator(camera[Camera::LEFT], camera[Camera::RIGHT],
                                  cameraPose[Camera::LEFT], cameraPose[Camera::RIGHT]);

        std::vector<CloudPoint> cvPointcloud;
        triangulator.triangulateCv(calculator.points1, calculator.points2, cvPointcloud);

        totalPoints.insert(totalPoints.end(), calculator.points1.begin(), calculator.points1.end());
        finalResult.insert(finalResult.end(), cvPointcloud.begin(), cvPointcloud.end());

//        Visualization matVis(cameraPose[Camera::LEFT], camera[Camera::LEFT]->cameraMatrix);
//        matVis.renderWithDepth(cvPointcloud);
//        imshow("Result", matVis.getResult());
//        waitKey();
    }


    Visualization matVis(cameraPose[Camera::LEFT], camera[Camera::LEFT]->cameraMatrix);
    matVis.renderWithDepth(finalResult);
    imshow("FinalResult", matVis.getResult());
    writeCloudPoints(finalResult);


//    namedWindow("Result", 1);
//
//    //imshow("magic", matVis.getResult());
//
//    CameraPose virtualPose;
//    Mat virtualCameraMatrix = (Mat_<double>(3, 3) << 540, 0, 320, 0, 540, 240, 0, 0, 1);
//    cameraPose[Camera::LEFT].copyTo(virtualPose);
//
//    int pos = 0;
//    char const *xTrackbar = "Váltás kamerák között";
//    createTrackbar(xTrackbar, "Result", &pos, 100);
//
//    VIS_TYPE type = VIS_TYPE::DEPTH;
//
//    while (true) {
//        // calculate the relative position.
//        double ratio = ((double) pos) / 100;
//        virtualPose.tvec = (1 - ratio) * cameraPose[Camera::LEFT].tvec + ratio * cameraPose[Camera::RIGHT].tvec;
//        virtualPose.rvec = slerp(cameraPose[Camera::LEFT].rvec, cameraPose[Camera::RIGHT].rvec, ratio);
//
//        Visualization matVis2(virtualPose, virtualCameraMatrix);
//        if (type == VIS_TYPE::DEPTH) {
//            matVis2.renderWithDepth(finalResult);
//        } else if (type == VIS_TYPE::PIXELS) {
//            matVis2.renderWithGrayscale(finalResult, totalPoints, frames[0]);
//        } else {
//            matVis2.renderWithContours(finalResult);
//        }
//        imshow("Result", matVis2.getResult());
//
//        char ch = (char) waitKey(33);
//        if (ch == 27) {
//            break;
//        } else if (ch == 'd') {
//            cameraPose[Camera::RIGHT].copyTo(virtualPose);
//            pos = 100;
//            setTrackbarPos(xTrackbar, "Result", pos);
//        } else if (ch == 'a') {
//            cameraPose[Camera::LEFT].copyTo(virtualPose);
//            pos = 0;
//            setTrackbarPos(xTrackbar, "Result", pos);
//        } else if (ch == '1') {
//            type = VIS_TYPE::DEPTH;
//        } else if (ch == '2') {
//            type = VIS_TYPE::PIXELS;
//        } else if (ch == '3') {
//            type = VIS_TYPE::CONTORUS;
//        } else if (ch == 'f') {
//            imwrite("/media/balint/Data/Linux/diploma/visualization.png",
//                    matVis2.getResult()(Rect(165, 145, 315, 315)));
//        }
//    }



//    while (true) {
    char ch = (char) waitKey();
//        if (ch == 27) {
//            break;
//        }
//    }

    return 0;
}
