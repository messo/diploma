#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui.hpp>
#include "OpticalFlowCalculator.h"
#include "Common.h"
#include "OFReconstruction.h"

using namespace cv;
using namespace std;


void OpticalFlowCalculator::feed(const Mat &image, const Rect &boundingRect, const vector<Point> &contour) {

    // --------------------------------------
    // 1. translate the image to the "center"
    // --------------------------------------

    Rect newBoundingRect(Point(320 - boundingRect.width / 2, 240 - boundingRect.height / 2), boundingRect.size());

    currentFrame.setTo(0);
    Point translation(newBoundingRect.tl() - boundingRect.tl());
    image(boundingRect).copyTo(currentFrame(newBoundingRect));

    // translating the contour

    currentContour.clear();
    translate(contour, translation, currentContour);

    if (!prevFrame.empty()) {
        // -----------------------------
        // 2.
        // -----------------------------

        this->calcOpticalFlow();
    }

    prevContour = currentContour;
    currentFrame.copyTo(prevFrame);
}

void OpticalFlowCalculator::calcOpticalFlow() {

    Mat flow;
    calcOpticalFlowFarneback(prevFrame, currentFrame, flow, 0.5, 4, 21, 10, 7, 1.5, OPTFLOW_FARNEBACK_GAUSSIAN);

    Mat backFlow;
    calcOpticalFlowFarneback(currentFrame, prevFrame, backFlow, 0.5, 4, 21, 10, 7, 1.5, OPTFLOW_FARNEBACK_GAUSSIAN);

    // this->visualizeOpticalFlow(flow);


    // trying to smooth the vectorfield
    //Mat smoothedFlow;
    //GaussianBlur(flow2, smoothedFlow, Size(15, 15), -1);
    //smoothedFlow.copyTo(flow2);

    // collecting matching points using optical flow
    vector<Point2f> points1, points2;
    this->collectMatchingPoints(flow, backFlow, points1, points2);
    this->visualizeMatches(points1, points2);

    // reconstruction
    currentReconstruction = Ptr<OFReconstruction>(new OFReconstruction(camera, points1, points2));
    currentReconstruction->reconstruct();
}

void OpticalFlowCalculator::collectMatchingPoints(const Mat &flow, const Mat &backFlow, vector<Point2f> &imgpts1, vector<Point2f> &imgpts2) {
    for (int y = 0; y < flow.rows; y++) {
        for (int x = 0; x < flow.cols; x++) {

            // from + fwd = to
            Point from(x, y);
            const Point2f &fwd = flow.at<Point2f>(y, x);
            Point to(cvRound(x + fwd.x), cvRound(y + fwd.y));

            if (pointPolygonTest(prevContour, from, false) > 0.0 &&
                    pointPolygonTest(currentContour, to, false) > 0.0) {

                // to + back = backTo
                const Point2f &back = backFlow.at<Point2f>(to.y, to.x);
                Point backTo(cvRound(to.x + back.x), cvRound(to.y + back.y));

                Point diff2(backTo.x - from.x, backTo.y - from.y);

                // only use if the backflow points almost there
                if (diff2.dot(diff2) <= 1) {

                    // don't use zero movements...
                    if (fwd.dot(fwd) <= 0.0001) {
                        continue;
                    }

                    imgpts1.push_back(Point2f(from));
                    imgpts2.push_back(Point2f(x + fwd.x, y + fwd.y));
                }

            }
        }
    }
}

// -----------------
//  VISUALIZATIONS
// -----------------

void OpticalFlowCalculator::visualizeOpticalFlow(const cv::Mat &flow) {
    Mat vis;
    currentFrame.copyTo(vis);
    cvtColor(vis, vis, cv::COLOR_GRAY2BGR);

    for (int y = 0; y < flow.rows; y += 5) {
        for (int x = 0; x < flow.cols; x += 5) {

            Point from(x, y);
            const Point2f &f = flow.at<Point2f>(y, x);
            Point to(cvRound(x + f.x), cvRound(y + f.y));

            if (pointPolygonTest(prevContour, from, false) > 0.0 &&
                    pointPolygonTest(currentContour, to, false) > 0.0) {

                line(vis, from, to, Scalar(0, 255, 0));
                circle(vis, from, 2, Scalar(255, 0, 0), -1);

            }
        }
    }

    imshow("opticalFlowVis", vis);
}

void OpticalFlowCalculator::visualizeMatches(const vector<Point2f> &points1, const vector<Point2f> &points2) {
    Mat vis;
    currentFrame.copyTo(vis);
    cvtColor(vis, vis, cv::COLOR_GRAY2BGR);

    for (int i = 0; i < points1.size(); i++) {
        Point p1(points1[i]);
        Point p2(points2[i]);

        if (p1.x % 5 == 0 && p1.y % 5 == 0) {
            line(vis, p1, p2, Scalar(0, 255, 0));
            circle(vis, p1, 2, Scalar(255, 0, 0), -1);
        }
    }

    imshow("matchesVis", vis);
}

void OpticalFlowCalculator::dumpReconstruction() {
    writeCloudPoints(currentReconstruction->resultingCloud);
}
