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

    Ptr<Camera> camera[2];
    camera[Camera::LEFT] = Ptr<Camera>(
            new RealCamera(Camera::LEFT, "/media/balint/Data/Linux/diploma/src/final/intrinsics_left.yml"));
    camera[Camera::RIGHT] = Ptr<Camera>(
            new RealCamera(Camera::RIGHT, "/media/balint/Data/Linux/diploma/src/final/intrinsics_right.yml"));

    CameraPose cameraPose[2];

    cameraPose[Camera::LEFT].load("/media/balint/Data/Linux/diploma/src/final/pose_left.yml");
    Matx34d leftP = cameraPose[Camera::LEFT].getProjectionMatrix();
    Matx44d leftPclP = cameraPose[Camera::LEFT].getPoseForPcl();
    cameraPose[Camera::RIGHT].load("/media/balint/Data/Linux/diploma/src/final/pose_right.yml");
    Matx34d rightP = cameraPose[Camera::RIGHT].getProjectionMatrix();
    Matx44d rightPclP = cameraPose[Camera::RIGHT].getPoseForPcl();

    Ptr<BackgroundSubtractorMOG2> bgSub[2];
    bgSub[Camera::LEFT] = createBackgroundSubtractorMOG2(300, 25.0, true);
    bgSub[Camera::RIGHT] = createBackgroundSubtractorMOG2(300, 25.0, true);

    SpatialOpticalFlowCalculator ofCalculator(camera[Camera::LEFT], camera[Camera::RIGHT]);

    int focus = 80;
    static_cast<RealCamera *>(camera[Camera::LEFT].get())->focus(focus);
    static_cast<RealCamera *>(camera[Camera::RIGHT].get())->focus(focus);

    ObjectSelector objSelector[2];

    double learningRate = -1;

    PclVisualization vis;

    while (true) {
        Mat selected[2];

        //double t0 = getTickCount();

#pragma omp parallel for
        for (int i = 0; i < 2; i++) {
            Mat image, mask;
            camera[i]->readUndistorted(image);
            bgSub[i]->apply(image, mask, learningRate);
            dilateAndErode(mask);
            //drawGridXY(leftImage, leftCamera, leftCameraPose);

            selected[i] = objSelector[i].selectUsingContoursWithMaxArea(image, mask);
        }

//        t0 = ((double) getTickCount() - t0) / getTickFrequency();
//        std::cout << "Done in " << t0 << "s" << std::endl;
//        std::cout.flush();

        imshow("leftImage", selected[Camera::LEFT]);
        imshow("rightImage", selected[Camera::RIGHT]);

        char ch = (char) waitKey(33);

        if (ch == 27) {
            break;
        }
        if (ch == 'o') {

            double t = getTickCount();

            ofCalculator.feed(selected, objSelector);

            std::cout << "## Feed done in " << (((double) getTickCount() - t) / getTickFrequency()) << "s" << std::endl;
            std::cout.flush();

            std::vector<CloudPoint> pointcloud;
            std::vector<Point> cp;
            TriangulatePoints(ofCalculator.points1, camera[Camera::LEFT]->K, camera[Camera::LEFT]->Kinv,
                              ofCalculator.points2, camera[Camera::RIGHT]->K, camera[Camera::RIGHT]->Kinv,
                              leftP, rightP, pointcloud, cp);

            t = ((double) getTickCount() - t) / getTickFrequency();
            std::cout << "## Done in " << t << "s" << std::endl;
            std::cout.flush();

            // VISU
            vis.init();
            vis.addCamera(camera[Camera::LEFT], leftPclP, 1);
            vis.addCamera(camera[Camera::RIGHT], rightPclP, 2);
            vis.addChessboard();
            vis.addPointCloud(pointcloud, 0);
            // waitKey();
        }

//        if (ch == 'w') {
//            focus += 1;
//            static_cast<RealCamera *>(leftCamera.get())->focus(focus);
//        } else if (ch == 's') {
//            focus -= 1;
//            static_cast<RealCamera *>(leftCamera.get())->focus(focus);
//        }
    }

    return 0;
}
