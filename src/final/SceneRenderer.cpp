#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include "camera/Camera.hpp"
#include "camera/DummyCamera.hpp"
#include "camera/CameraPose.h"
#include "mask/ForegroundMaskCalculator.h"
#include "mask/OFForegroundMaskCalculator.h"
#include "object/Matcher.h"
#include "object/MultiObjectSelector.h"
#include "optical_flow/OpticalFlowCalculator.h"
#include "Common.h"
#include "Triangulator.h"
#include "Visualization.h"
#include "PerformanceMonitor.h"
#include "mask/MOG2ForegroundMaskCalculator.h"

using namespace cv;
using namespace std;

int main(int argc, char **argv) {

    cv::cuda::DeviceInfo devInfo;
    cv::cuda::setDevice(devInfo.deviceID());

    omp_set_nested(1);

    vector<Ptr<Camera>> camera(2);
    DummyCamera *cam1 = new DummyCamera(Camera::LEFT, "/media/balint/Data/Linux/diploma/scene_3/scene", 620);
    cam1->readCalibration("/media/balint/Data/Linux/diploma/src/final/intrinsics_left.yml");
    camera[Camera::LEFT] = Ptr<Camera>(cam1);
    DummyCamera *cam2 = new DummyCamera(Camera::RIGHT, "/media/balint/Data/Linux/diploma/scene_3/scene", 620);
    cam2->readCalibration("/media/balint/Data/Linux/diploma/src/final/intrinsics_right.yml");
    camera[Camera::RIGHT] = Ptr<Camera>(cam2);

    vector<CameraPose> cameraPose(2);
    cameraPose[Camera::LEFT].load("/media/balint/Data/Linux/diploma/scene_3/polc_pose_left.yml");
    cameraPose[Camera::RIGHT].load("/media/balint/Data/Linux/diploma/scene_3/polc_pose_right.yml");

    std::vector<Ptr<ForegroundMaskCalculator>> maskCalculators(2);
    maskCalculators[Camera::LEFT] = Ptr<ForegroundMaskCalculator>(new OFForegroundMaskCalculator());
    maskCalculators[Camera::RIGHT] = Ptr<ForegroundMaskCalculator>(new OFForegroundMaskCalculator());

    FileStorage fs;
    fs.open("/media/balint/Data/Linux/diploma/scene_3/F.yml", FileStorage::READ);
    Mat F;
    fs["myF"] >> F;

    Matcher matcher = Matcher(camera[Camera::LEFT], camera[Camera::RIGHT], F);
    MultiObjectSelector objSelector(matcher);

    int count = 0;

//    double ratio = 0.5;
//    CameraPose virtualPose;
//    virtualPose.tvec = (1 - ratio) * cameraPose[Camera::LEFT].tvec + ratio * cameraPose[Camera::RIGHT].tvec;
//    virtualPose.rvec = slerp(cameraPose[Camera::LEFT].rvec, cameraPose[Camera::RIGHT].rvec, ratio);
//    Mat virtualCameraMatrix = (Mat_<double>(3, 3) << 540, 0, 320, 0, 540, 240, 0, 0, 1);

    while (cam1->frame != cam1->lastFrame) {

        count++;

        if (count == 3) {
            // RESET PerformanceIndicator
            PerformanceMonitor::get()->reset();
        }

        PerformanceMonitor::get()->frameStarted();

        std::cout << "======== NEXT FRAME ===== " << std::endl;
        double t0 = getTickCount();

        std::vector<std::vector<Mat>> selected = getFramesFromCameras(camera, maskCalculators);
        std::vector<Mat> &frames = selected[0];
        std::vector<Mat> &masks = selected[1];

        std::vector<Object> objects = objSelector.selectObjects(frames, masks);

        std::vector<CloudPoint> finalResult;
        std::vector<Point2f> totalPoints;

        PerformanceMonitor::get()->objFound(objects.size());

        std::vector<std::vector<CloudPoint>> clouds(objects.size());
        std::vector<std::vector<Point2f>> points(objects.size());

#pragma omp parallel for
        for (int i = 0; i < objects.size(); i++) {
            OpticalFlowCalculator calculator;
            std::pair<std::vector<Point2f>, std::vector<Point2f>> matches = calculator.calcDenseMatches(frames, objects[i]);

            PerformanceMonitor::get()->triangulationStarted();

            Triangulator triangulator(camera[Camera::LEFT], camera[Camera::RIGHT],
                                      cameraPose[Camera::LEFT], cameraPose[Camera::RIGHT]);

            double reproj = triangulator.triangulateCv(matches.first, matches.second, clouds[i]);
            PerformanceMonitor::get()->reprojError(reproj);
            points[i] = matches.first;

            PerformanceMonitor::get()->triangulationFinished();
        }

        if (objects.size() > 0) {
            PerformanceMonitor::get()->objectBasedStuff();
        }

        // collecting results...
        for (int i = 0; i < objects.size(); i++) {
            totalPoints.insert(totalPoints.end(), points[i].begin(), points[i].end());
            finalResult.insert(finalResult.end(), clouds[i].begin(), clouds[i].end());
        }

        PerformanceMonitor::get()->visualizationStarted();
//        Visualization matVis(virtualPose, virtualCameraMatrix);
        Visualization matVis(cameraPose[0], camera[0]->cameraMatrix);
        matVis.renderWithDepth(finalResult);

//        matVis.renderWithContours(clouds);

        PerformanceMonitor::get()->visualizationFinished();


//            imshow("frame1", frames[0]);
//            imshow("frame2", frames[1]);
//
//        imshow("result", matVis.getResult());
//        waitKey();

        imwrite("/media/balint/Data/Linux/diploma/__out/vis_" + to_string(cam1->frame - 1) + ".png", matVis.getResult());
//
//        imwrite("/media/balint/Data/Linux/diploma/scene_1_tiny_vis/vis_" + to_string(cam1->frame - 1) + "_left.png", frames[0]);
//        imwrite("/media/balint/Data/Linux/diploma/scene_1_tiny_vis/vis_" + to_string(cam1->frame - 1) + "_right.png", frames[1]);

        std::cout << "FRAME FINISHED IN: " << (getTickCount() - t0) / getTickFrequency() << " s" << endl;
        if (objects.size() > 0) {
            PerformanceMonitor::get()->frameFinished();
        }
    }

    std::cout << "################" << std::endl;
    PerformanceMonitor::get()->reproj.print("AVG");
    PerformanceMonitor::get()->frameTotal.print("Frame");
    PerformanceMonitor::get()->maskCalculation.print("MaskCalculation");
    PerformanceMonitor::get()->extracting.print("Matching/Features");
    PerformanceMonitor::get()->objSelection.print("Matching/ObjSelection");

//    PerformanceMonitor::get()->ofInit.print("OF/Init/Object");
    PerformanceMonitor::get()->ofInitPerFrame.print("OF/Init/Frame");
//    PerformanceMonitor::get()->ofCalc.print("OF/Calculation/Object");
    PerformanceMonitor::get()->ofCalcPerFrame.print("OF/Calculation/Frame");
//    PerformanceMonitor::get()->ofMatching.print("OF/PointMatching/Object");
    PerformanceMonitor::get()->ofMatchingPerFrame.print("OF/PointMatching/Frame");
//    PerformanceMonitor::get()->triangulation.print("Triangulate/Object");
    PerformanceMonitor::get()->triangulationPerFrame.print("Triangulate/Frame");

    PerformanceMonitor::get()->visualization.print("Visualization");

    std::cout << "################" << std::endl;
    for (auto it = PerformanceMonitor::get()->objectCount.begin(); it != PerformanceMonitor::get()->objectCount.end(); ++it) {
        std::cout << "Objects: " << it->first << " count: " << it->second << std::endl;
    }

    return 0;
}
