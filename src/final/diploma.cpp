#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include "camera/RealCamera.hpp"
#include "calibration/Calibration.h"
#include "camera/CameraPose.h"
#include "calibration/CameraPoseCalculator.h"

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
            calculator.drawGrid(image);
        }
        imshow("image", image);

        char ch = (char) waitKey(33);

        if (ch == 27) {
            break;
        } else if (ch == 'p') {
            if(calculator.calculate()) {
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

void drawBoxOnChessboard(Mat inputImage, Ptr<Camera> camera, Ptr<CameraPose> pose) {
    // coordinates for box
    std::vector<Point3f> objectPoints;
    objectPoints.push_back(Point3f(0, 0, 0));
    objectPoints.push_back(Point3f(0, 8, 0));
    objectPoints.push_back(Point3f(5, 8, 0));
    objectPoints.push_back(Point3f(5, 0, 0));

    objectPoints.push_back(Point3f(0, 0, 5));
    objectPoints.push_back(Point3f(0, 8, 5));
    objectPoints.push_back(Point3f(5, 8, 5));
    objectPoints.push_back(Point3f(5, 0, 5));

    // calculating imagePoints
    std::vector<Point2f> imagePoints;
    projectPoints(objectPoints, pose->rvec, pose->tvec, camera->K, camera->distCoeffs, imagePoints);

    // drawing
    line(inputImage, imagePoints[0], imagePoints[1], Scalar(0, 0, 255), 1);
    line(inputImage, imagePoints[1], imagePoints[2], Scalar(0, 0, 255), 1);
    line(inputImage, imagePoints[2], imagePoints[3], Scalar(0, 0, 255), 1);
    line(inputImage, imagePoints[3], imagePoints[0], Scalar(0, 0, 255), 1);

    line(inputImage, imagePoints[4], imagePoints[5], Scalar(0, 0, 255), 1);
    line(inputImage, imagePoints[5], imagePoints[6], Scalar(0, 0, 255), 1);
    line(inputImage, imagePoints[6], imagePoints[7], Scalar(0, 0, 255), 1);
    line(inputImage, imagePoints[7], imagePoints[4], Scalar(0, 0, 255), 1);

    line(inputImage, imagePoints[0], imagePoints[4], Scalar(0, 0, 255), 1);
    line(inputImage, imagePoints[1], imagePoints[5], Scalar(0, 0, 255), 1);
    line(inputImage, imagePoints[2], imagePoints[6], Scalar(0, 0, 255), 1);
    line(inputImage, imagePoints[3], imagePoints[7], Scalar(0, 0, 255), 1);
}


int main(int argc, char **argv) {

//    calibrate(Camera::LEFT);
//    calibrate(Camera::RIGHT);

//    calcPose(Camera::LEFT, "intrinsics_left.yml", "pose_left.yml");
//    calcPose(Camera::RIGHT, "intrinsics_right.yml", "pose_right.yml");



    return 0;
}
