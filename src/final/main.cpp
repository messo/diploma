#include <iostream>

#include <opencv2/video/background_segm.hpp>
#include <opencv2/video/tracking.hpp>
#include <GL/freeglut_std.h>

#define FPS_ENABLED

#include "Calibration.hpp"
#include "StereoCamera.hpp"
#include "OpenGLRenderer.hpp"
#include "DinoCamera.hpp"
#include "DummyCamera.hpp"
#include "ObjectSelector.hpp"
#include "OFReconstruction.h"

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

Mat mergeImages(const Mat &left, const Mat &right) {
    Mat result = Mat::zeros(left.rows, left.cols + right.cols, left.type());

    left.copyTo(result(Rect(0, 0, left.cols, left.rows)));
    right.copyTo(result(Rect(left.cols - 1, 0, right.cols, right.rows)));

    return result;
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
    Ptr<DummyCamera> camera(new DummyCamera(Camera::LEFT, "/media/balint/Data/Linux/diploma/src/imgs_stereo_depth_single", 302));
    camera->readCalibration("/media/balint/Data/Linux/diploma/src/imgs_stereo_calibration/intrinsics.yml");

    //Ptr<DinoCamera> camera(new DinoCamera());

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

    //VideoWriter outputVideo;
    //outputVideo.open("/media/balint/Data/Linux/diploma/src/magic.mkv", VideoWriter::fourcc('X', 'V', 'I', 'D'), 10.0, Size(1280, 480), true);

    // előző kép lekérdezése
    Mat prevGray, img, mask;

    std::vector<cv::Point> prevLastContour;

    Ptr<ObjectSelector> objSelector(new ObjectSelector());

    int i = 1;

    while (true) {
        // aktuális kép...
        camera->retrieve(img);

        bgSub->apply(img, mask, 0);
        // objektum kiválasztása...

        int niters = 3;
        dilate(mask, mask, Mat(), Point(-1, -1), niters);
        erode(mask, mask, Mat(), Point(-1, -1), niters * 2);
        dilate(mask, mask, Mat(), Point(-1, -1), niters);

        //mask = Mat::ones(480, 640, CV_8U);

        //erodeAndDilate(mask, 2);
        //dilateAndErode(mask, 25);
        //erode(mask, 5);

        // --- CONNECTED COMPONENTS
        Mat selected(objSelector->selectUsingContoursWithClosestCentroid(img, mask));
        imshow("selected", selected);
        // ------------------------

        /*char filename1[80];
        sprintf(filename1,"/media/balint/Data/Linux/selected/%d.png",i++);
        imwrite(filename1, selected);*/

        Mat selectedGray;
        cvtColor(selected, selectedGray, cv::COLOR_BGR2GRAY);

        if (!prevGray.empty()) {
            Mat flow;
            calcOpticalFlowFarneback(prevGray, selectedGray, flow, 0.25, 4, 13, 10, 5, 1.1, 0);

            Mat backFlow;
            calcOpticalFlowFarneback(selectedGray, prevGray, backFlow, 0.25, 4, 13, 10, 5, 1.1, 0);

            Mat tmp, vis;
            prevGray.copyTo(tmp);
            cvtColor(tmp, vis, cv::COLOR_GRAY2BGR);

            Mat flow2(flow.rows, flow.cols, flow.type());

            for (int y = 0; y < flow.rows; y++) {
                for (int x = 0; x < flow.cols; x++) {

                    const Point2f &fxy = flow.at<Point2f>(y, x);
                    Point from(x, y);
                    Point to(cvRound(x + fxy.x), cvRound(y + fxy.y));

                    double length2 = fxy.dot(fxy);

                    if (pointPolygonTest(prevLastContour, from, false) > 0.0 &&
                            pointPolygonTest(objSelector->getLastContour(), to, false) > 0.0 /*&&
                            length2 > MIN_SQ_LENGTH && length2 < MAX_SQ_LENGTH*/) {
                        flow2.at<Point2f>(y, x) = fxy;
                    } else {
                        flow2.at<Point2f>(y, x) = Point2f(0.0, 0.0);
                    }
                }
            }



            // trying to smooth the vectorfield
            Mat smoothedFlow;
            GaussianBlur(flow2, smoothedFlow, Size(15, 15), -1);
            smoothedFlow.copyTo(flow2);



            vector<Point2f> imgpts1, imgpts2;

            for (int y = 0; y < flow2.rows; y += 3) {
                for (int x = 0; x < flow2.cols; x += 3) {

                    const Point2f &fxy = flow2.at<Point2f>(y, x);
                    Point from(x, y);
                    Point to(cvRound(x + fxy.x), cvRound(y + fxy.y));

                    if (fxy.x != 0.0 && fxy.y != 0.0) {
                        Point2f avg(getSparseAverage(flow2, x, y));
                        Point to2(cvRound(x + avg.x), cvRound(y + avg.y));

                        Point diff(to2.x - to.x, to2.y - to.y);

                        if (diff.dot(diff) <= 1) {
                            const Point2f &back = backFlow.at<Point2f>(to.y, to.x);
                            Point backTo(cvRound(to.x + back.x), cvRound(to.y + back.y));

                            Point diff2(backTo.x - from.x, backTo.y - from.y);
                            if(diff2.dot(diff2) <= 1) {
                                imgpts1.push_back(Point2f(from));
                                imgpts2.push_back(Point2f(x + fxy.x, y + fxy.y));
                                line(vis, from, to, Scalar(0, 255, 0));

                                //const Point2f &s = smoothedFlow.at<Point2f>(y, x);
                                //line(vis, from, Point(cvRound(x + s.x), cvRound(y + s.y)), Scalar(0, 0, 255));

                                circle(vis, from, 2, Scalar(255, 0, 0), -1);
                            } else {
                                //line(vis, from, to, Scalar(0, 0, 255));
                                //line(vis, to, backTo, Scalar(255, 0, 0));
                            }
                        } else {
                            //line(vis, from, to, Scalar(0, 0, 255));
                            //circle(vis, from, 2, Scalar(0, 0, 255), -1);
                        }
                    }
                }
            }

            OFReconstruction reconstruction(camera, imgpts1, imgpts2);
            reconstruction.reconstruct();

            imshow("vis", vis);
            //Mat merged(mergeImages(img, vis));
            //imshow("merged", merged);

            //outputVideo << merged;
        }

        selectedGray.copyTo(prevGray);
        prevLastContour = objSelector->getLastContour();

        char key = waitKey();
        if (key == 27) {
            //outputVideo.release();
            return 0;
        }
    }
}
