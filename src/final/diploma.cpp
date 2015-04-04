#include <opencv2/highgui.hpp>
#include "camera/RealCamera.hpp"
#include "calibration/Calibration.h"
#include "camera/CameraPose.h"
#include "calibration/CameraPoseCalculator.h"
#include "Common.h"

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

    Ptr<Camera> camera(new RealCamera(Camera::LEFT, "intrinsics_left.yml"));
    Ptr<CameraPose> cameraPose(new CameraPose());
    cameraPose->load("pose_left.yml");

    while (true) {
        Mat image;
        camera->readUndistorted(image);
        drawGridXY(image, camera, cameraPose);
        imshow("image", image);

        char ch = (char) waitKey(33);

        if (ch == 27) {
            break;
        }
    }

    return 0;
}
