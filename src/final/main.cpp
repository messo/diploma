#include <iostream>

#include <opencv2/video/background_segm.hpp>
#include <opencv2/video/tracking.hpp>
#include <GL/freeglut_std.h>

#define FPS_ENABLED

#include "StereoCalibration.hpp"
#include "StereoCamera.hpp"
#include "OpenGLRenderer.hpp"
#include "DinoCamera.hpp"
#include "DummyCamera.hpp"
#include "ObjectSelector.hpp"
#include "OFReconstruction.h"
#include "Common.h"
#include "OpticalFlowCalculator.h"
#include "Triangulation.h"
#include "MultiView.h"
#include "PclVisualization.h"

#define DEPTH_ENABLED true

#define MIN_SQ_LENGTH 0.01
#define MAX_SQ_LENGTH 10000.0

using namespace std;
using namespace cv;

void clearImage(Mat mat, Rect rect) {
    rectangle(mat, rect.tl(), rect.br(), Scalar(0, 0, 0), -1, LINE_8, 0);
}

void dilateAndErode(Mat img, int erosion_size) {
    Mat kernel = getStructuringElement(MORPH_RECT,
                                       Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                       Point(erosion_size, erosion_size));

    dilate(img, img, kernel);
    erode(img, img, kernel);
}

void erodeAndDilate(Mat img, int erosion_size) {
    Mat kernel = getStructuringElement(MORPH_RECT,
                                       Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                       Point(erosion_size, erosion_size));

    erode(img, img, kernel);
    dilate(img, img, kernel);
}

void erode(Mat img, int erosion_size) {
    Mat kernel = getStructuringElement(MORPH_RECT,
                                       Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                       Point(erosion_size, erosion_size));

    erode(img, img, kernel);
}

Point2f getDenseAverage(Mat flow, int x, int y) {

    int size = 5;

    Point2f sum;
    int count = 0;

    for (int i = x - size; i <= x + size; i++) {
        for (int j = y - size; j <= y + size; j++) {
            const Point2f &p = flow.at<Point2f>(j, x);
            if (p.x != 0.0 && p.y != 0.0) {
                sum.x += p.x;
                sum.y += p.y;
                count++;
            }
        }
    }

    return Point2f(sum.x / count, sum.y / count);
}

Point2f getSparseAverage(Mat flow, int x, int y) {

    int size = 5;

    Point2f sum;
    int count = 0;

    for (int i = x - size; i <= x + size; i += size) {
        for (int j = y - size; j <= y + size; j += size) {
            const Point2f &p = flow.at<Point2f>(j, x);
            if (p.x != 0.0 && p.y != 0.0) {
                sum.x += p.x;
                sum.y += p.y;
                count++;
            }
        }
    }

    return Point2f(sum.x / count, sum.y / count);
}

int main(int argc, char **argv) {
    Ptr<DummyCamera> camera(
            new DummyCamera(Camera::LEFT, "/media/balint/Data/Linux/diploma/src/imgs_stereo_depth_single", 302));
    camera->readCalibration("/media/balint/Data/Linux/diploma/src/imgs_stereo_calibration/intrinsics.yml");

    //Ptr<DinoCamera> camera(new DinoCamera());

    cout << "Building background model..." << endl;

    /// ----- MODELL ÉPÍTÉS!!!
    Ptr<BackgroundSubtractorMOG2> bgSub = createBackgroundSubtractorMOG2(302, 25.0, false);
    // dummy part, build a proper model, this cannot be used in real-time magic!
    for (int i = 0; i < 500; i++) {
        Mat current, mask;
        camera->retrieve(current);
        bgSub->apply(current, mask);
    }

    camera->firstFrame = 60;
    camera->reset();
    /// ------ MODELL KÉSZ

    cout << "Background model done." << endl;

    //VideoWriter outputVideo;
    //outputVideo.open("/media/balint/Data/Linux/diploma/src/magic.mkv", VideoWriter::fourcc('X', 'V', 'I', 'D'), 10.0, Size(1280, 480), true);

    // előző kép lekérdezése
    Mat prevGray, img, mask;

    std::vector<cv::Point> prevLastContour;

    Ptr<ObjectSelector> objSelector(new ObjectSelector());

    OpticalFlowCalculator calc(camera);
    MultiView mv(camera);

    PclVisualization pcl;

    bool initialStructureExists = false;

    while (true) {
        // aktuális kép...
        camera->retrieve(img);

        // maszk meghatározása és finomítása...
        bgSub->apply(img, mask, 0);
        int niters = 3;
        dilate(mask, mask, Mat(), Point(-1, -1), niters);
        erode(mask, mask, Mat(), Point(-1, -1), niters * 2);
        dilate(mask, mask, Mat(), Point(-1, -1), niters);

        // B opció, ami elég fostos...
        //mask = Mat::ones(480, 640, CV_8U);
        //erodeAndDilate(mask, 2);
        //dilateAndErode(mask, 25);
        //erode(mask, 5);

        // objektum kiválasztása + szürkítés
        Mat _selected(objSelector->selectUsingContoursWithClosestCentroid(img, mask));
        Mat _selectedGray;
        cvtColor(_selected, _selectedGray, cv::COLOR_BGR2GRAY);

        /*imshow("current", _selectedGray);

        char key1 = waitKey();
        if(key1 != 'o') {
            continue;
        }*/

        // fő algoritmus etetése az új képpel és egyéb infókkal

        bool ofResult = calc.feed(camera->getFrameId(), _selectedGray, objSelector->lastMask,
                                  objSelector->lastBoundingRect,
                                  objSelector->getLastContour());

        // ofResult mutatja, hogy "használható-e a dolog"
        if (ofResult) {
            // két esetünk van, nincs még rekonstrukciónk, tehát kell csinálni egyet, vagy már van és ki kell egészíteni a dolgot.
            if (!initialStructureExists) {
                std::cout << "BUILDING INITITAL STRUCTURE: prevFrame: " << calc.prevFrameId << " currentFrame: " <<
                calc.currentFrameId << std::endl;

                Ptr<OFReconstruction> reconstruction(
                        new OFReconstruction(camera, calc.prevFrameId, calc.points1, calc.currentFrameId,
                                             calc.points2));
                reconstruction->reconstruct();

                mv.addP(calc.prevFrameId, reconstruction->P1);
                mv.addP(calc.currentFrameId, reconstruction->P2);

                // FIXME -- do we need to implement = operator???
                mv.cloud = reconstruction->resultingCloud;
                // writeCloudPoints("cloud0.ply", mv.cloud.points);

                pcl.addCamera(camera, mv.P(calc.prevFrameId), calc.currentFrameId);
                pcl.addPointCloud(mv.cloud.points, calc.currentFrameId);

                // visualize
//                Mat vis(480, 640, CV_8UC3, Scalar(0, 0, 0));
//                std::map<int, Point2i> &from = cloud.lookup2DByIdx[60];
//                std::map<int, Point2i> &to = cloud.lookup2DByIdx[63];
//                for (int i = 0; i < cloud.points.size(); i++) {
//                    Point2i &p1 = from[i];
//                    Point2i &p2 = to[i];
//
//                    if (p1.x % 5 == 0 && p1.y % 5 == 0) {
//                        line(vis, p1, p2, Scalar(0, 255, 0));
//                        circle(vis, p1, 2, Scalar(255, 0, 0), -1);
//                    }
//                }
//                imshow("VISmagic", vis);

                initialStructureExists = true;
            } else {
                // új kép -> elmozdulások alapján egyeződés, ebből camera pose!

                mv.reconstructNext(calc.prevFrameId, calc.points1, calc.currentFrameId, calc.points2);

                pcl.addCamera(camera, mv.P(calc.currentFrameId), calc.currentFrameId);
                pcl.addPointCloud(mv.cloud.points, calc.currentFrameId);
            }
        }

        char key = waitKey();
        if (key == 27) {
            //outputVideo.release();
            return 0;
        }
    }
}
