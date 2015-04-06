#include <opencv2/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/imgproc.hpp>
#include "camera/RealCamera.hpp"
#include "camera/CameraPose.h"
#include "calibration/Calibration.h"
#include "calibration/CameraPoseCalculator.h"
#include "optical_flow/SpatialOpticalFlowCalculator.h"
#include "ObjectSelector.hpp"
#include "camera/DummyCamera.hpp"
#include "Triangulation.h"
#include "PclVisualization.h"

using namespace cv;

void calibrate(int cameraId) {
    Ptr<Camera> camera(new RealCamera(cameraId));
    Calibration calibration(camera);

    while (true) {

        Mat img;
        camera->read(img);

        imshow("img", img);

        char ch = (char) waitKey(33);

        if (ch == ' ') {
            if (calibration.acquireFrames()) {
                Mat chessboard;
                calibration.drawChessboardCorners(chessboard);
                imshow("img", chessboard);
                waitKey(500);
            } else {
                std::cout << "Finding chessboard failed. Please acquire a new frame." << std::endl;
            }
        } else if (ch == 'c') {
            if (calibration.calibrate()) {
                calibration.save((cameraId == Camera::LEFT) ? "intrinsics_left.yml" : "intrinsics_right.yml");
                std::cout << "Calibration succeeded." << std::endl;
            } else {
                std::cout << "Calibration failed." << std::endl;
            }
        } else if (ch == 27) {
            break;
        }
    }
}

void calcPose(int cameraId, const std::string &calibrationFile, const std::string &poseFile) {
    Ptr<Camera> camera(new RealCamera(cameraId, calibrationFile));
    CameraPoseCalculator calculator(camera);

    while (true) {
        Mat image;
        camera->readUndistorted(image);
        if (calculator.poseCalculated()) {
            drawGridXY(image, camera, calculator.cameraPose);
            drawBoxOnChessboard(image, camera, calculator.cameraPose);
        }
        imshow("image", image);

        char ch = (char) waitKey(33);

        if (ch == 27) {
            break;
        } else if (ch == 'p') {
            if (calculator.calculate()) {
                std::cout << "Calculated." << std::endl;
            } else {
                std::cout << "Not calculated!" << std::endl;
            }
        }
    }

    if (calculator.poseCalculated()) {
        calculator.cameraPose->save(poseFile);
    }
}

void dilateAndErode(Mat mask) {
    int niters = 3;
    dilate(mask, mask, Mat(), Point(-1, -1), niters);
    erode(mask, mask, Mat(), Point(-1, -1), niters * 2);
    dilate(mask, mask, Mat(), Point(-1, -1), niters);
}

int main(int argc, char **argv) {

//    calibrate(Camera::LEFT);
//    calibrate(Camera::RIGHT);

//    calcPose(Camera::LEFT, "/media/balint/Data/Linux/diploma/src/final/intrinsics_left.yml", "pose_left.yml");
//    calcPose(Camera::RIGHT, "/media/balint/Data/Linux/diploma/src/final/intrinsics_right.yml", "pose_right.yml");

//    Ptr<Camera> leftBgCamera(new DummyCamera(Camera::LEFT, "/media/balint/Data/Linux/diploma/src/bg", 27));
    Ptr<Camera> leftCamera(
            new RealCamera(Camera::LEFT, "/media/balint/Data/Linux/diploma/src/final/intrinsics_left.yml"));
    Ptr<CameraPose> leftCameraPose(new CameraPose());
    leftCameraPose->load("/media/balint/Data/Linux/diploma/src/final/pose_left.yml");
    Ptr<BackgroundSubtractorMOG2> leftBgSub = createBackgroundSubtractorMOG2(300, 25.0, true);

//    Ptr<Camera> rightBgCamera(new DummyCamera(Camera::RIGHT, "/media/balint/Data/Linux/diploma/src/bg", 27));
    Ptr<Camera> rightCamera(
            new RealCamera(Camera::RIGHT, "/media/balint/Data/Linux/diploma/src/final/intrinsics_right.yml"));
    Ptr<CameraPose> rightCameraPose(new CameraPose());
    rightCameraPose->load("/media/balint/Data/Linux/diploma/src/final/pose_right.yml");
    Ptr<BackgroundSubtractorMOG2> rightBgSub = createBackgroundSubtractorMOG2(300, 25.0, true);

    SpatialOpticalFlowCalculator ofCalculator(leftCamera, rightCamera);

    int focus = 80;
    static_cast<RealCamera *>(leftCamera.get())->focus(focus);
    static_cast<RealCamera *>(rightCamera.get())->focus(focus);

    ObjectSelector leftObjSelector;
    ObjectSelector rightObjSelector;

    double learningRate = -1;

    // BUILD MODEL
//    for(int i=0; i<28; i++) {
//        Mat leftImage, rightImage, mask;
//
//        leftBgCamera->read(leftImage);
//        rightBgCamera->read(rightImage);
//
//        leftBgSub->apply(leftImage, mask);
//        rightBgSub->apply(rightImage, mask);
//    }

//    int i = 0;

    PclVisualization vis;

    while (true) {
        Mat leftImage, leftMask;
        leftCamera->readUndistorted(leftImage);
        leftBgSub->apply(leftImage, leftMask, learningRate);
        dilateAndErode(leftMask);
        //drawGridXY(leftImage, leftCamera, leftCameraPose);

        Mat leftSelected(leftObjSelector.selectUsingContoursWithMaxArea(leftImage, leftMask));

        imshow("leftImage", leftSelected);

        Mat rightImage, rightMask;
        rightCamera->readUndistorted(rightImage);
        rightBgSub->apply(rightImage, rightMask, learningRate);
        dilateAndErode(rightMask);
        //drawGridXY(rightImage, rightCamera, rightCameraPose);

        Mat rightSelected(rightObjSelector.selectUsingContoursWithMaxArea(rightImage, rightMask));

        imshow("rightImage", rightSelected);
//        imshow("rightMask", rightMask);

//        char filename1[80];
//        sprintf(filename1,"/media/balint/Data/Linux/diploma/src/bg/left_%d.png",i);
//        imwrite(filename1, leftImage);
//
//        char filename2[80];
//        sprintf(filename2,"/media/balint/Data/Linux/diploma/src/bg/right_%d.png",i);
//        imwrite(filename2, rightImage);
//        i++;

        char ch = (char) waitKey(33);

        if (ch == 27) {
            break;
        }
        if (ch == 'o') {

            double t = getTickCount();

            ofCalculator.feed(leftSelected, leftObjSelector.lastMask,
                              rightSelected, rightObjSelector.lastMask);

            Matx33d leftR;
            Rodrigues(leftCameraPose->rvec, leftR);
            Matx34d leftP = Matx34d(leftR(0, 0), leftR(0, 1), leftR(0, 2), leftCameraPose->tvec.at<double>(0, 0),
                                    leftR(1, 0), leftR(1, 1), leftR(1, 2), leftCameraPose->tvec.at<double>(1, 0),
                                    leftR(2, 0), leftR(2, 1), leftR(2, 2), leftCameraPose->tvec.at<double>(2, 0));

            Matx33d rightR;
            Rodrigues(rightCameraPose->rvec, rightR);
            Matx34d rightP = Matx34d(rightR(0, 0), rightR(0, 1), rightR(0, 2), rightCameraPose->tvec.at<double>(0, 0),
                                     rightR(1, 0), rightR(1, 1), rightR(1, 2), rightCameraPose->tvec.at<double>(1, 0),
                                     rightR(2, 0), rightR(2, 1), rightR(2, 2), rightCameraPose->tvec.at<double>(2, 0));

            std::vector<CloudPoint> pointcloud;
            std::vector<Point> cp;
            TriangulatePoints(ofCalculator.points1, leftCamera->K, leftCamera->Kinv, ofCalculator.points2,
                              rightCamera->K, rightCamera->Kinv, leftP, rightP, pointcloud, cp);


            // VISU!!!

            t = ((double) getTickCount() - t) / getTickFrequency();
            std::cout << "Done in " << t << "s" << std::endl;
            std::cout.flush();

            vis.init();

            Mat leftViewMatrix = Mat::zeros(4, 4, CV_64F);
            Mat rightViewMatrix = Mat::zeros(4, 4, CV_64F);
            for (unsigned int row = 0; row < 3; ++row) {
                for (unsigned int col = 0; col < 3; ++col) {
                    leftViewMatrix.at<double>(row, col) = leftR(row, col);
                    rightViewMatrix.at<double>(row, col) = rightR(row, col);
                }
                leftViewMatrix.at<double>(row, 3) = leftCameraPose->tvec.at<double>(row, 0);
                rightViewMatrix.at<double>(row, 3) = rightCameraPose->tvec.at<double>(row, 0);
            }
            leftViewMatrix.at<double>(3, 3) = 1.0;
            rightViewMatrix.at<double>(3, 3) = 1.0;

            Mat cvToGl = Mat::zeros(4, 4, CV_64F);
            cvToGl.at<double>(0, 0) = 1.0;
            cvToGl.at<double>(1, 1) = 1.0; // Invert the y axis
            cvToGl.at<double>(2, 2) = -1.0; // invert the z axis
            cvToGl.at<double>(3, 3) = 1.0;

            vis.addCamera(leftCamera, cvToGl * leftViewMatrix, 1);
            vis.addCamera(rightCamera, cvToGl * rightViewMatrix, 2);

            vis.addChessboard();
            vis.addPointCloud(pointcloud, 0);

            waitKey();
        }
        if (ch == 'w') {
            focus += 1;
            static_cast<RealCamera *>(leftCamera.get())->focus(focus);
        } else if (ch == 's') {
            focus -= 1;
            static_cast<RealCamera *>(leftCamera.get())->focus(focus);
        }
    }

    return 0;
}
