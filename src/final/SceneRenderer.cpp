#include <opencv2/highgui.hpp>
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

using namespace cv;
using namespace std;

int main(int argc, char **argv) {

    vector<Ptr<Camera>> camera(2);
    DummyCamera *cam1 = new DummyCamera(Camera::LEFT, "/media/balint/Data/Linux/diploma/scene_1", 228);
    cam1->readCalibration("/media/balint/Data/Linux/diploma/src/final/intrinsics_left.yml");
    camera[Camera::LEFT] = Ptr<Camera>(cam1);
    DummyCamera *cam2 = new DummyCamera(Camera::RIGHT, "/media/balint/Data/Linux/diploma/scene_1", 228);
    cam2->readCalibration("/media/balint/Data/Linux/diploma/src/final/intrinsics_right.yml");
    camera[Camera::RIGHT] = Ptr<Camera>(cam2);

    vector<CameraPose> cameraPose(2);
    cameraPose[Camera::LEFT].load("/media/balint/Data/Linux/diploma/src/final/pose_left.yml");
    cameraPose[Camera::RIGHT].load("/media/balint/Data/Linux/diploma/src/final/pose_right.yml");

    std::vector<Ptr<ForegroundMaskCalculator>> maskCalculators(2);
    maskCalculators[Camera::LEFT] = Ptr<ForegroundMaskCalculator>(new OFForegroundMaskCalculator());
    maskCalculators[Camera::RIGHT] = Ptr<ForegroundMaskCalculator>(new OFForegroundMaskCalculator());

    FileStorage fs;
    fs.open("/media/balint/Data/Linux/diploma/F.yml", FileStorage::READ);
    Mat F;
    fs["myF"] >> F;

    Matcher matcher = Matcher(camera[Camera::LEFT], camera[Camera::RIGHT], F);
    MultiObjectSelector objSelector(matcher);

    OpticalFlowCalculator calculator;

    while (cam1->frame != cam1->lastFrame) {

        PerformanceMonitor::get()->frameStarted();

        std::cout << "======== NEXT FRAME ===== " << std::endl;
        double t0 = getTickCount();

        std::vector<std::vector<Mat>> selected = getFramesFromCameras(camera, maskCalculators);
        std::vector<Mat> &frames = selected[0];
        std::vector<Mat> &masks = selected[1];

        std::vector<Object> objects = objSelector.selectObjects(frames, masks);

        std::vector<CloudPoint> finalResult;
        std::vector<Point2f> totalPoints;

        //imshow("frame1", frames[0]);
        //imshow("frame2", frames[1]);

        PerformanceMonitor::get()->objFound(objects.size());

        for (int i = 0; i < objects.size(); i++) {
            std::pair<std::vector<Point2f>, std::vector<Point2f>> matches = calculator.calcDenseMatches(frames, objects[i]);

            PerformanceMonitor::get()->triangulationStarted();

            Triangulator triangulator(camera[Camera::LEFT], camera[Camera::RIGHT],
                                      cameraPose[Camera::LEFT], cameraPose[Camera::RIGHT]);

            std::vector<CloudPoint> cvPointcloud;
            triangulator.triangulateCv(matches.first, matches.second, cvPointcloud);

            totalPoints.insert(totalPoints.end(), matches.first.begin(), matches.first.end());
            finalResult.insert(finalResult.end(), cvPointcloud.begin(), cvPointcloud.end());

            PerformanceMonitor::get()->triangulationFinished();
        }

        if (objects.size() > 0) {
            PerformanceMonitor::get()->objectBasedStuff();
        }

        PerformanceMonitor::get()->visualizationStarted();
        Visualization matVis(cameraPose[Camera::LEFT], camera[Camera::LEFT]->cameraMatrix);
        matVis.renderWithDepth(finalResult);
        PerformanceMonitor::get()->visualizationFinished();

        imwrite("/media/balint/Data/Linux/diploma/scene_1/vis_" + to_string(cam1->frame - 1) + ".png", matVis.getResult());

        std::cout << "FRAME FINISHED IN: " << (getTickCount() - t0) / getTickFrequency() << " s" << endl;
        PerformanceMonitor::get()->frameFinished();
    }

    std::cout << "################" << std::endl;
    PerformanceMonitor::get()->frameTotal.print("Frame");
    PerformanceMonitor::get()->maskCalculation.print("MaskCalculation");
    PerformanceMonitor::get()->extracting.print("Matching/Features");
    PerformanceMonitor::get()->objSelection.print("Matching/ObjSelection");

    PerformanceMonitor::get()->ofInit.print("OF/Init/Object");
    PerformanceMonitor::get()->ofInitPerFrame.print("OF/Init/Frame");
    PerformanceMonitor::get()->ofCalc.print("OF/Calculation/Object");
    PerformanceMonitor::get()->ofCalcPerFrame.print("OF/Calculation/Frame");
    PerformanceMonitor::get()->ofMatching.print("OF/PointMatching/Object");
    PerformanceMonitor::get()->ofMatchingPerFrame.print("OF/PointMatching/Frame");
    PerformanceMonitor::get()->triangulation.print("Triangulate/Object");
    PerformanceMonitor::get()->triangulationPerFrame.print("Triangulate/Frame");

    PerformanceMonitor::get()->visualization.print("Visualization");

    std::cout << "################" << std::endl;
    for (auto it = PerformanceMonitor::get()->objectCount.begin(); it != PerformanceMonitor::get()->objectCount.end(); ++it) {
        std::cout << "Objects: " << it->first << " count: " << it->second << std::endl;
    }

    return 0;
}
