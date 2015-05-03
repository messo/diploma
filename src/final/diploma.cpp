#include <opencv2/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/imgproc.hpp>
#include <iomanip>
#include "camera/RealCamera.hpp"
#include "camera/CameraPose.h"
#include "calibration/Calibration.h"
#include "calibration/CameraPoseCalculator.h"
#include "optical_flow/OpticalFlowCalculator.h"
#include "object/SingleObjectSelector.hpp"
#include "camera/DummyCamera.hpp"
#include "Triangulator.h"
#include "locking.h"
#include "Visualization.h"
#include "FPSCounter.h"
#include "mask/MOG2ForegroundMaskCalculator.h"
#include "mask/OFForegroundMaskCalculator.h"
#include "object/MultiObjectSelector.h"

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
            if (calibration.acquireFrame()) {
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

int counter = 0;

int main(int argc, char **argv) {

//    calibrate(Camera::LEFT);
//    calibrate(Camera::RIGHT);

//    calcPose(Camera::LEFT, "/media/balint/Data/Linux/diploma/src/final/intrinsics_left.yml", "pose_left.yml");
//    calcPose(Camera::RIGHT, "/media/balint/Data/Linux/diploma/src/final/intrinsics_right.yml", "pose_right.yml");

    std::vector<Ptr<Camera>> camera(2);
    camera[Camera::LEFT] = Ptr<Camera>(
            new RealCamera(Camera::LEFT, "/media/balint/Data/Linux/diploma/src/final/intrinsics_left.yml"));
    camera[Camera::RIGHT] = Ptr<Camera>(
            new RealCamera(Camera::RIGHT, "/media/balint/Data/Linux/diploma/src/final/intrinsics_right.yml"));

    std::vector<CameraPose> cameraPose(2);
    cameraPose[Camera::LEFT].load("/media/balint/Data/Linux/diploma/src/final/pose_left.yml");
    Matx34d leftP = cameraPose[Camera::LEFT].getRT();
    //Matx44d leftPclP = cameraPose[Camera::LEFT].getPoseForPcl();
    cameraPose[Camera::RIGHT].load("/media/balint/Data/Linux/diploma/src/final/pose_right.yml");
    Matx34d rightP = cameraPose[Camera::RIGHT].getRT();
    //Matx44d rightPclP = cameraPose[Camera::RIGHT].getPoseForPcl();

    std::vector<Ptr<ForegroundMaskCalculator>> maskCalculators(2);
    maskCalculators[Camera::LEFT] = Ptr<ForegroundMaskCalculator>(
            new OFForegroundMaskCalculator()); //new MOG2ForegroundMaskCalculator());
    maskCalculators[Camera::RIGHT] = Ptr<ForegroundMaskCalculator>(
            new OFForegroundMaskCalculator()); //new MOG2ForegroundMaskCalculator());

    FileStorage fs;
    fs.open("/media/balint/Data/Linux/diploma/F.yml", FileStorage::READ);
    Mat F;
    fs["myF"] >> F;
    OpticalFlowCalculator ofCalculator;

    int focus = 80;
    static_cast<RealCamera *>(camera[Camera::LEFT].get())->focus(focus);
    static_cast<RealCamera *>(camera[Camera::RIGHT].get())->focus(focus);

    Matcher matcher(camera[Camera::LEFT], camera[Camera::RIGHT], F);
    Ptr<ObjectSelector> objSelector(new SingleObjectSelector(matcher));

    Triangulator triangulator(camera[Camera::LEFT], camera[Camera::RIGHT],
                              cameraPose[Camera::LEFT], cameraPose[Camera::RIGHT]);

    double learningRate;

    // PclVisualization vis;
    Visualization matVis(cameraPose[Camera::LEFT], camera[Camera::LEFT]->cameraMatrix);
    Visualization matVis2(cameraPose[Camera::LEFT], camera[Camera::LEFT]->cameraMatrix);

    bool taskRunning = false;

    // Data to share
    MutexType mutex;
    Mat frame0, frame1;
    Mat mask0, mask1;

    long frameCounter = 0;

    omp_set_nested(1);

#pragma omp parallel
    {
#pragma omp single
        {
            FPSCounter dispCnter, procCnter;

            while (true) {
                //double t0 = getTickCount();

                frameCounter++;

//                if (frameCounter < 500) {
                learningRate = 0.001;
//                } else {
                //std::cout << "LEARNING OFF" << std::endl;
//                    learningRate = 0;
//                }

                std::vector<std::vector<Mat>> selected = getFramesFromCameras(camera, maskCalculators);
                std::vector<Mat> &frames = selected[0];
                std::vector<Mat> &masks = selected[1];

                Mat leftRight = mergeImages(frames[Camera::LEFT], frames[Camera::RIGHT]);
                dispCnter.tick();
                // std::cout << "Display: " << dispCnter.get() << std::endl;
                imshow("input", leftRight);

                char ch = (char) waitKey(1);
                if (ch == 27) {
                    break;
                } else if (ch == ' ') {
                    taskRunning = false;
                }

                // copy the current data if it's needed.
                mutex.Lock();
                frame0 = frames[0].clone();
                frame1 = frames[1].clone();
                mask0 = masks[0].clone();
                mask1 = masks[1].clone();

#pragma omp task firstprivate(frame0,frame1, mask0, mask1)
                {
                    mutex.Unlock();
                    if (!taskRunning) {
                        taskRunning = true;

                        std::vector<Mat> frames(2);
                        frames[0] = frame0;
                        frames[1] = frame1;

                        std::vector<Mat> masks(2);
                        masks[0] = mask0;
                        masks[1] = mask1;

                        std::vector<Object> objects = objSelector->selectObjects(frames, masks);
                        if (objects.size() != 0) {

                            std::cout << "---------------------------------" << std::endl;

                            std::vector<CloudPoint> totalCloud;

                            // TODO MORE OBJECTS!!
                            for (int i = 0; i < objects.size(); i++) {
                                if (boundingRect(objects[i].masks[0]).area() > 100 && boundingRect(objects[i].masks[1]).area() > 100) {
                                    double t = getTickCount();

                                    std::pair<std::vector<Point2f>, std::vector<Point2f>> matches = ofCalculator.calcDenseMatches(frames, objects[i]);

                                    std::cout << "[" << std::setw(20) << "main" << "] " << "Object(" << i << ") feed done in " <<
                                    (((double) getTickCount() - t) / getTickFrequency()) << "s" << std::endl;
                                    std::cout.flush();

//                        std::vector<CloudPoint> pointcloud;
//                        std::vector<Point> cp;
//                        TriangulatePoints(ofCalculator.points1, camera[Camera::LEFT]->cameraMatrix,
//                                          camera[Camera::LEFT]->Kinv,
//                                          ofCalculator.points2, camera[Camera::RIGHT]->cameraMatrix,
//                                          camera[Camera::RIGHT]->Kinv,
//                                          leftP, rightP, pointcloud, cp);

                                    std::vector<CloudPoint> cvPointcloud;
                                    triangulator.triangulateCv(matches.first, matches.second, cvPointcloud);
                                    totalCloud.insert(totalCloud.end(), cvPointcloud.begin(), cvPointcloud.end());
                                }
                            }

//                        matVis.renderWithDepth(pointcloud);
                            matVis2.renderWithDepth(totalCloud);
                        }

//                        imwrite("__left.png", frame0);
//                        imwrite("__right.png", frame1);
//                        imwrite("__mask_left.png", mask0);
//                        imwrite("__mask_right.png", mask1);

// VISU
//                        vis.init();
//                        vis.addCamera(camera[Camera::LEFT], leftPclP, 1);
//                        vis.addCamera(camera[Camera::RIGHT], rightPclP, 2);
//                        vis.addChessboard();
//                        vis.addPointCloud(pointcloud, 0);

                        taskRunning = false;
                    }
                }

//                imshow("magic", matVis.getResult());
                imshow("magicCV", matVis2.getResult());
            }
        }
    }

    return 0;
}
