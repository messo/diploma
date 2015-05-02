#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Geometry>
#include "camera/Camera.hpp"
#include "camera/RealCamera.hpp"
#include "calibration/CameraPoseCalculator.h"
#include "Common.h"
#include "optical_flow/ReportOpticalFlowCalculator.h"
#include "Triangulator.h"
#include "Visualization.h"
#include "object/SingleObjectSelector.hpp"
#include "mask/OFForegroundMaskCalculator.h"
#include "mask/MOG2ForegroundMaskCalculator.h"
#include "optical_flow/SpatialOpticalFlowCalculator.h"
#include "object/MultiObjectSelector.h"

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

int main___maskcalc(int argc, char **argv) {

    Ptr<Camera> camera1(new RealCamera(Camera::LEFT, "/media/balint/Data/Linux/diploma/src/final/intrinsics_left.yml"));
    Ptr<Camera> camera2(new RealCamera(Camera::RIGHT, "/media/balint/Data/Linux/diploma/src/final/intrinsics_right.yml"));

    OFForegroundMaskCalculator maskCalculator1, maskCalculator2;

    int frameId = 0;
    while (true) {

        frameId++;

        Mat frame1, frame2;
        camera1->read(frame1);
        Mat mask1 = maskCalculator1.calculate(frame1);

        camera2->read(frame2);
        Mat mask2 = maskCalculator2.calculate(frame2);

        imshow("frame1", frame1);
        imshow("mask1", mask1);
        imshow("frame2", frame2);
        imshow("mask2", mask2);

        char ch = (char) waitKey(33);
        if (ch == 27) {
            break;
        } else if (frameId > 200) {
            imwrite("/media/balint/Data/Linux/frames/frame_left_" + std::to_string(frameId) + ".png", frame1);
            imwrite("/media/balint/Data/Linux/frames/frame_right_" + std::to_string(frameId) + ".png", frame2);
            imwrite("/media/balint/Data/Linux/frames/mask_left_" + std::to_string(frameId) + ".png", mask1);
            imwrite("/media/balint/Data/Linux/frames/mask_right_" + std::to_string(frameId) + ".png", mask2);
        }
    }

    return 0;
}

int main_SELECT(int argc, char **argv) {

    Ptr<Camera> camera1(new RealCamera(Camera::LEFT, "/media/balint/Data/Linux/diploma/src/final/intrinsics_left.yml"));
    Ptr<Camera> camera2(new RealCamera(Camera::RIGHT, "/media/balint/Data/Linux/diploma/src/final/intrinsics_right.yml"));

    FileStorage fs;
    fs.open("/media/balint/Data/Linux/diploma/F.yml", FileStorage::READ);
    Mat F;
    fs["myF"] >> F;

    Matcher matcher(camera1, camera2, F);
    SingleObjectSelector objectSelector(matcher);

    std::vector<Mat> frames(2), masks(2);

    frames[0] = imread("/media/balint/Data/Linux/mask/frame_ofmask_104.png");
    frames[1] = imread("/media/balint/Data/Linux/mask/frame_ofmask_107.png");
    masks[0] = imread("/media/balint/Data/Linux/mask/mask_ofmask_104.png", IMREAD_GRAYSCALE);
    masks[1] = imread("/media/balint/Data/Linux/mask/mask_ofmask_107.png", IMREAD_GRAYSCALE);

    std::vector<Object> objects = objectSelector.selectObjects(frames, masks);

    imwrite("/media/balint/Data/Linux/mask/frame_ofmask_104_selected.png", objects[0].masks[0]);

    int frameId = 0;
    while (true) {

        imshow("masks", objects[0].masks[0]);

        char ch = (char) waitKey(33);
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

Mat getFundamentalMat(const Ptr<Camera> &camera1, const CameraPose &pose1,
                      const Ptr<Camera> &camera2, const CameraPose &pose2) {
    Mat R1, R2;
    Mat T1 = pose1.tvec;
    Mat T2 = pose2.tvec;
    Rodrigues(pose1.rvec, R1);
    Rodrigues(pose2.rvec, R2);

    Mat R = R2 * R1.inv();
    Mat T = T2 - R * T1;

    Matx33d Tx(0, -T.at<double>(2), T.at<double>(1),
               T.at<double>(2), 0, -T.at<double>(0),
               -T.at<double>(1), T.at<double>(0), 0);

    Mat E = Mat(Tx) * R;

    return Mat(camera2->cameraMatrix.t().inv()) * E * Mat(camera1->cameraMatrix.inv());
}

int main_doublePoseWithF(int argc, char **argv) {

    Ptr<Camera> camera1(new RealCamera(Camera::LEFT, "/media/balint/Data/Linux/diploma/src/final/intrinsics_left.yml"));
    CameraPoseCalculator calculator1(camera1);
    Ptr<Camera> camera2(
            new RealCamera(Camera::RIGHT, "/media/balint/Data/Linux/diploma/src/final/intrinsics_right.yml"));
    CameraPoseCalculator calculator2(camera2);

    int count = 0;
    bool undistorted = false;
    while (true) {
        Mat image1, image2;

        if (undistorted) {
            camera1->readUndistorted(image1);
            camera2->readUndistorted(image2);
        } else {
            camera1->read(image1);
            camera2->read(image2);
        }

        if (calculator1.poseCalculated()) {
            drawGridXY(image1, camera1, calculator1.cameraPose);
            // drawBoxOnChessboard(image, camera, calculator.cameraPose);

//            Size boardSize(9, 6);
//            float squareSize = 1.0f;
//
//            std::vector<Point3f> objectPoints;
//            for (int k = 0; k < boardSize.height; k++)
//                for (int j = 0; j < boardSize.width; j++)
//                    objectPoints.push_back(Point3f(squareSize * j, squareSize * k, 0));
//
//            Mat imagePoints;
//            projectPoints(objectPoints, calculator1.cameraPose->rvec, calculator1.cameraPose->tvec, camera1->cameraMatrix, camera1->distCoeffs, imagePoints);
//
//            for(int i=0; i<9*6; i++) {
//                circle(image1, imagePoints.at<Point2f>(i), 1, Scalar(255, 255, 255), 1);
//            }
        }
        if (calculator2.poseCalculated()) {
            drawGridXY(image2, camera2, calculator2.cameraPose);
            // drawBoxOnChessboard(image, camera, calculator.cameraPose);
        }

        imshow("image1", image1);
        imshow("image2", image2);

        char ch = (char) waitKey(33);
        count++;

        if (ch == 27) {
            break;
        } else if (ch == 'p') {
            if (calculator1.calculate() && calculator2.calculate()) {

                calculator1.cameraPose->save("pose_left.yml");
                calculator2.cameraPose->save("pose_right.yml");

                FileStorage fs;
                fs.open("F.yml", FileStorage::WRITE);

                Mat points1, points2;
                undistortPoints(calculator1.imagePoints, points1, camera1->cameraMatrix, camera1->distCoeffs, Mat(),
                                camera1->cameraMatrix);
                undistortPoints(calculator2.imagePoints, points2, camera2->cameraMatrix, camera2->distCoeffs, Mat(),
                                camera2->cameraMatrix);

                Point2f x1 = points1.at<Point2f>(0);
                Point2f x2 = points2.at<Point2f>(0);

                // try#1
                Mat F = findFundamentalMat(points1, points2);

                // try#2

                // CALCULATING ESSENTIAL MATRIX
                Mat F2 = getFundamentalMat(camera1, *calculator1.cameraPose, camera2, *calculator2.cameraPose);

                fs << "cvF" << F << "myF" << F2;

//                std::cout << F2 << endl;
//                std::cout << Mat(Matx13d(x2.x, x2.y, 1)) * F2 * Mat(Matx31d(x1.x, x1.y, 1)) << endl;
//
//
//                // trying to use
//
//                Mat newPoints1, newPoints2;
//                correctMatches(F, points1, points2, newPoints1, newPoints2);
//
//                Point2f newx1 = newPoints1.at<Point2f>(0);
//                Point2f newx2 = newPoints2.at<Point2f>(0);
//
//                std::cout << x1 << " " << newx1 << endl;
//
//                std::cout << Mat(Matx13d(newx2.x, newx2.y, 1)) * F * Mat(Matx31d(newx1.x, newx1.y, 1)) << endl;
//
//                // F2
//
//                Mat newPoints1_, newPoints2_;
//                correctMatches(F2, points1, points2, newPoints1_, newPoints2_);
//
//                Point2f newx1_ = newPoints1_.at<Point2f>(0);
//                Point2f newx2_ = newPoints2_.at<Point2f>(0);
//
//                std::cout << newx1 << " " << newx1_ << endl;
//
//                std::cout << Mat(Matx13d(newx2_.x, newx2_.y, 1)) * F2 * Mat(Matx31d(newx1_.x, newx1_.y, 1)) << endl;

                std::cout << "Calculated." << std::endl;
            } else {
                std::cout << "Not calculated!" << std::endl;
            }
        } else if (ch == 'u') {
            undistorted = !undistorted;
        }
    }

    return 0;
}

int main_SAVE(int argc, char **argv) {
    vector<Ptr<Camera>> camera(2);
    camera[Camera::LEFT] = Ptr<Camera>(
            new RealCamera(Camera::LEFT, "/media/balint/Data/Linux/diploma/src/final/intrinsics_left.yml"));
    camera[Camera::RIGHT] = Ptr<Camera>(
            new RealCamera(Camera::RIGHT, "/media/balint/Data/Linux/diploma/src/final/intrinsics_right.yml"));

    while (true) {
        Mat left, right;

        camera[0]->read(left);
        camera[1]->read(right);

        imshow("left", left);
        imshow("right", right);

        char ch = (char) waitKey(33);
        if (ch == 27) {
            break;
        } else if (ch == ' ') {
            imwrite("left.png", left);
            imwrite("right.png", right);
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

    FileStorage fs;
    fs.open("/media/balint/Data/Linux/diploma/F.yml", FileStorage::READ);
    Mat F;
    fs["myF"] >> F;

    OFForegroundMaskCalculator maskCalculator1, maskCalculator2;
    Matcher matcher(camera[0], camera[1], F);
    MultiObjectSelector objSelector(matcher);
    SpatialOpticalFlowCalculator ofCalculator(camera[Camera::LEFT], camera[Camera::RIGHT], F);

    int frame = 0;

    while (true) {
        frame++;
        std::vector<Mat> frames(2), masks(2);

        camera[0]->read(frames[0]);
        masks[0] = maskCalculator1.calculate(frames[0]);

        camera[1]->read(frames[1]);
        masks[1] = maskCalculator2.calculate(frames[1]);

        vector<Object> objects = objSelector.selectObjects(frames, masks);

//        if (boundingRect(objects[0].masks[0]).area() < 100 || boundingRect(objects[0].masks[1]).area() < 100) {
//            continue;
//        }
//
//        ofCalculator.feed(frames, objects[0]);
//
//        Triangulator triangulator(camera[Camera::LEFT], camera[Camera::RIGHT],
//                                  cameraPose[Camera::LEFT], cameraPose[Camera::RIGHT]);
//        std::vector<CloudPoint> cvPointcloud;
//        triangulator.triangulateCv(ofCalculator.points1, ofCalculator.points2, cvPointcloud);
//
//        Visualization matVis2(cameraPose[Camera::LEFT], camera[Camera::LEFT]->cameraMatrix);
//        matVis2.renderWithDepth(cvPointcloud);
//        imshow("Result", matVis2.getResult());

        imshow("left", frames[0]);
        imshow("right", frames[1]);


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

        char ch = (char) waitKey(33);
        if (ch == 27) {
            break;
        } else if (frame % 5 == 0) {
            imwrite("/media/balint/Data/Linux/for_multi/" + std::to_string(frame) + "_1left_img.png", frames[0]);
            imwrite("/media/balint/Data/Linux/for_multi/" + std::to_string(frame) + "_2left_mask.png", masks[0]);
            imwrite("/media/balint/Data/Linux/for_multi/" + std::to_string(frame) + "_3right_img.png", frames[1]);
            imwrite("/media/balint/Data/Linux/for_multi/" + std::to_string(frame) + "_4right_mask.png", masks[1]);
            imwrite("/media/balint/Data/Linux/for_multi/" + std::to_string(frame) + "_5objects.png", merged);
//            imwrite("/media/balint/Data/Linux/for_multi/" + std::to_string(frame) + "_5vis.png", matVis2.getResult());
        }
    }
}

enum VIS_TYPE {
    PIXELS, DEPTH, CONTOURS
};

int main_________(int argc, char **argv) {

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

    FileStorage fs;
    fs.open("/media/balint/Data/Linux/diploma/F.yml", FileStorage::READ);
    Mat F;
    fs["myF"] >> F;

    Matcher matcher(camera[0], camera[1], F);
    SingleObjectSelector objSelector(matcher);


    SpatialOpticalFlowCalculator ofCalculator(camera[Camera::LEFT], camera[Camera::RIGHT], F);

    Mat originalImage = imread("/media/balint/Data/Linux/diploma/of_img_left.png");

    Mat left = imread("/media/balint/Data/Linux/for_report/390_1left_img.png", IMREAD_GRAYSCALE); // of_img_left
    Mat right = imread("/media/balint/Data/Linux/for_report/390_3right_img.png", IMREAD_GRAYSCALE); // of_img_right
    Mat maskLeft = imread("/media/balint/Data/Linux/for_report/390_2left_mask.png", IMREAD_GRAYSCALE);
    Mat maskRight = imread("/media/balint/Data/Linux/for_report/390_4right_mask.png", IMREAD_GRAYSCALE);

//    Mat eqLeft, eqRight;
//    equalizeHist(left, eqLeft);
//    equalizeHist(right, eqRight);

    // SURF
//    vector<Mat> images(2);
//    images[0] = left;
//    images[1] = right;
//
//    SURFFeatureExtractor extractor(images);
//
//    std::vector<DMatch> matches;
//    MyMatcher matcher(camera[0], camera[1]);
//    std::vector<std::pair<Point2f, Point2f>> points = matcher.match(extractor, matches, F);
//
//    //-- Draw matches
//    Mat img_matches;
//    drawMatches(images[0], extractor.keypoints[0], images[1], extractor.keypoints[1], matches, img_matches,
//                Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//
//    //-- Show detected matches
//    imshow("Matches", img_matches);
//    // -----------
//
//    // MAGIC
//    std::vector<Point2f> vectors;
//    for (int i = 0; i < points.size(); i++) {
//        vectors.push_back(points[i].second - points[i].first);
//    }
//    Point2f v = magicVector(vectors);
//
//
//
//    Rect bRect(boundingRect(maskLeft));
//    Mat shiftedLeft(480, 640, CV_8UC3);
//    shiftImage(left, bRect, Point2i(v.x, v.y), shiftedLeft);
//    imwrite("/media/balint/Data/Linux/diploma/__left_shifted.png", shiftedLeft);


    vector<Mat> frames(2);
    frames[0] = left;
    frames[1] = right;

    vector<Mat> masks(2);
    masks[0] = maskLeft;
    masks[1] = maskRight;

    vector<Object> objects = objSelector.selectObjects(frames, masks);

    ofCalculator.feed(frames, objects[0]);

    Triangulator triangulator(camera[Camera::LEFT], camera[Camera::RIGHT],
                              cameraPose[Camera::LEFT], cameraPose[Camera::RIGHT]);

    std::vector<CloudPoint> pointcloud;
    triangulator.triangulateIteratively(ofCalculator.points1, ofCalculator.points2, pointcloud);

    std::vector<CloudPoint> cvPointcloud;
    triangulator.triangulateCv(ofCalculator.points1, ofCalculator.points2, cvPointcloud);

//    Visualization matVis(cameraPose[Camera::LEFT], camera[Camera::LEFT]->cameraMatrix);

//    matVis.renderWithDepth(pointcloud);

    namedWindow("Result", 1);

    //imshow("magic", matVis.getResult());

    CameraPose virtualPose;
    Mat virtualCameraMatrix = (Mat_<double>(3, 3) << 540, 0, 320, 0, 540, 240, 0, 0, 1);
    cameraPose[Camera::LEFT].copyTo(virtualPose);

    int pos = 0;
    char const *xTrackbar = "Váltás kamerák között";
    createTrackbar(xTrackbar, "Result", &pos, 100);

    VIS_TYPE type = VIS_TYPE::DEPTH;

    while (true) {
        // calculate the relative position.
        double ratio = ((double) pos) / 100;
        virtualPose.tvec = (1 - ratio) * cameraPose[Camera::LEFT].tvec + ratio * cameraPose[Camera::RIGHT].tvec;
        virtualPose.rvec = slerp(cameraPose[Camera::LEFT].rvec, cameraPose[Camera::RIGHT].rvec, ratio);

        Visualization matVis2(virtualPose, virtualCameraMatrix);
        if (type == VIS_TYPE::DEPTH) {
            matVis2.renderWithDepth(cvPointcloud);
        } else if (type == VIS_TYPE::PIXELS) {
            matVis2.renderWithColors(cvPointcloud, ofCalculator.points1, originalImage);
        } else {
            matVis2.renderWithContours(cvPointcloud);
        }
        imshow("Result", matVis2.getResult());

        char ch = (char) waitKey(33);
        if (ch == 27) {
            break;
        } else if (ch == 'd') {
            cameraPose[Camera::RIGHT].copyTo(virtualPose);
            pos = 100;
            setTrackbarPos(xTrackbar, "Result", pos);
        } else if (ch == 'a') {
            cameraPose[Camera::LEFT].copyTo(virtualPose);
            pos = 0;
            setTrackbarPos(xTrackbar, "Result", pos);
        } else if (ch == '1') {
            type = VIS_TYPE::DEPTH;
        } else if (ch == '2') {
            type = VIS_TYPE::PIXELS;
        } else if (ch == '3') {
            type = VIS_TYPE::CONTOURS;
        } else if (ch == 'f') {
            imwrite("/media/balint/Data/Linux/diploma/visualization.png",
                    matVis2.getResult()(Rect(165, 145, 315, 315)));
        }
    }

    return 0;
}
