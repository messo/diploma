#include <opencv2/highgui.hpp>
#include "camera/RealCamera.hpp"
#include "camera/CameraPose.h"
#include "calibration/Calibration.h"
#include "calibration/CameraPoseCalculator.h"
#include "optical_flow/SpatialOpticalFlowCalculator.h"

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
                calibration.save(cameraId == Camera::LEFT ? "intrinsics_left.yml" : "intrinsics_right.yml");
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
    Ptr<Camera> camera(new RealCamera(cameraId));
    camera->readCalibration(calibrationFile);

    CameraPoseCalculator calculator(camera);

    while (true) {
        Mat image;
        camera->readUndistorted(image);
        if (calculator.poseCalculated()) {
            drawGridXY(image, camera, calculator.cameraPose);
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

int main(int argc, char **argv) {

//    calibrate(Camera::LEFT);
//    calibrate(Camera::RIGHT);

//    calcPose(Camera::LEFT, "intrinsics_left.yml", "pose_left.yml");
//    calcPose(Camera::RIGHT, "intrinsics_right.yml", "pose_right.yml");

    Ptr<Camera> leftCamera(new RealCamera(Camera::LEFT, "intrinsics_left.yml"));
    Ptr<CameraPose> lefCameraPose(new CameraPose());
    lefCameraPose->load("pose_left.yml");

    Ptr<Camera> rightCamera(new RealCamera(Camera::RIGHT, "intrinsics_right.yml"));
    Ptr<CameraPose> rightCameraPose(new CameraPose());
    rightCameraPose->load("pose_right.yml");

    SpatialOpticalFlowCalculator ofCalculator(leftCamera, rightCamera);

    int focus = 80;
    static_cast<RealCamera *>(leftCamera.get())->focus(focus);
    static_cast<RealCamera *>(rightCamera.get())->focus(focus);

    while (true) {
        Mat leftImage;
        leftCamera->readUndistorted(leftImage);
        //drawGridXY(leftImage, leftCamera, lefCameraPose);
        imshow("leftImage", leftImage);

        Mat rightImage;
        rightCamera->readUndistorted(rightImage);
        //drawGridXY(rightImage, rightCamera, rightCameraPose);
        imshow("rightImage", rightImage);

        ofCalculator.feed(leftImage, Mat(leftImage.rows, leftImage.cols, CV_8U, Scalar(255)),
                          rightImage, Mat(rightImage.rows, rightImage.cols, CV_8U, Scalar(255)));

        char ch = (char) waitKey(33);

        if (ch == 27) {
            break;
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
