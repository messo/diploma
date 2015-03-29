#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui.hpp>
#include "OpticalFlowCalculator.h"
#include "Common.h"
#include "OFReconstruction.h"

#define EPSILON_TEXTURE 3.0

using namespace cv;
using namespace std;


void OpticalFlowCalculator::feed(const Mat &image, const Mat &lastMask, const Rect &boundingRect,
                                 const vector<Point> &contour) {

    // --------------------------------------
    // 1. translate the image to the "center"
    // --------------------------------------

    Point translation((320 - boundingRect.width / 2) - boundingRect.x,
                      (240 - boundingRect.height / 2) - boundingRect.y);

    currentBoundingRect = boundingRect + translation;

    // translate the image
    shiftImage(image, boundingRect, translation, currentFrame);
    // translate the mask
    shiftImage(lastMask, boundingRect, translation, currentMask);
    // translating the contour
    currentContour.clear();
    translate(contour, translation, currentContour);

    // calculating textured regions
    this->calcTexturedRegions();

    double length;

    if (prevFrame.empty() || (length = this->calcOpticalFlow()) > 1.0) {
        // -----------------------------
        // 2.
        // -----------------------------

        std::cout << length << std::endl;
        std::cout.flush();

        prevContour = currentContour;
        currentFrame.copyTo(prevFrame);
        currentMask.copyTo(prevMask);
        currentTexturedRegions.copyTo(prevTexturedRegions);
    }
}

double OpticalFlowCalculator::calcOpticalFlow() {

    Mat flow;
    calcOpticalFlowFarneback(prevFrame, currentFrame, flow, 0.5, 4, 21, 10, 7, 1.5, OPTFLOW_FARNEBACK_GAUSSIAN);

    //this->getFlowAvg(flow);

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

    Point2f avgMovement(this->calcAverageMovement(points1, points2));

    double length = norm(avgMovement);
    cout << avgMovement << ": " << length << endl;
    cout.flush();

    if (length > 10.0) {
        // we consider this as "too big" displacement, so we shift the current image further, and do this again

        Point translation(-avgMovement);

        // translate the image
        Mat _currentFrame;
        currentFrame.copyTo(_currentFrame);
        shiftImage(_currentFrame, currentBoundingRect, translation, currentFrame);
        // translate the mask
        Mat _currentMask;
        currentMask.copyTo(_currentMask);
        shiftImage(_currentMask, currentBoundingRect, translation, currentMask);
        // translate the textured regions
        Mat _currentTexturedRegions;
        currentTexturedRegions.copyTo(_currentTexturedRegions);
        shiftImage(_currentTexturedRegions, currentBoundingRect, translation, currentTexturedRegions);
        // translating the contour
        translate(currentContour, translation);

        length = this->calcOpticalFlow();
    } else if (length >= 1.0) {
        this->visualizeMatches(points1, points2);

        // reconstruction
        currentReconstruction = Ptr<OFReconstruction>(new OFReconstruction(camera, points1, points2));
        currentReconstruction->reconstruct();
    } else {
        imshow("prev", prevFrame);
        imshow("this", currentFrame);
    }

    return length;
}

void OpticalFlowCalculator::calcTexturedRegions() {
    // H(σ(Iy - Ix) - ε)
    int ksize = 1;

    currentTexturedRegions.setTo(0);

    // FIXME -- this can be done in boundringRect!

    for (int y = 0; y < currentFrame.rows; y++) {
        for (int x = 0; x < currentFrame.cols; x++) {
            if (!currentMask.at<char>(y, x)) {
                continue;
            }

            char Ixy = currentFrame.at<char>(y, x);

            vector<char> values;
            for (int j = y - ksize; j <= y + ksize; j++) {
                for (int i = x - ksize; i <= x + ksize; i++) {
                    if ((i == x && j == y) || i < 0 || j < 0 || i >= currentFrame.cols || j >= currentFrame.rows)
                        continue;

                    if (currentMask.at<char>(j, i)) {
                        values.push_back(currentFrame.at<char>(j, i) - Ixy);
                    }
                }
            }

            if (values.size() == 0)
                continue;

            // for each neighbour
            long sum = 0;
            for (int k = 0; k < values.size(); k++) {
                sum += values[k];
            }
            double mu = (double) sum / values.size();

            double tmp = 0;
            for (int k = 0; k < values.size(); k++) {
                tmp += (values[k] - mu) * (values[k] - mu);
            }

            double sigma2 = tmp / values.size();

            if (sigma2 > EPSILON_TEXTURE) {
                currentTexturedRegions.at<unsigned char>(y, x) = 255;
            }
        }
    }
}

void OpticalFlowCalculator::collectMatchingPoints(const Mat &flow, const Mat &backFlow, vector<Point2f> &imgpts1,
                                                  vector<Point2f> &imgpts2) {
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

                    if (prevTexturedRegions.at<char>(from) && currentTexturedRegions.at<char>(to)) {
                        imgpts1.push_back(Point2f(from));
                        imgpts2.push_back(Point2f(x + fwd.x, y + fwd.y));
                    }
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

/*void OpticalFlowCalculator::getFlowAvg(const cv::Mat &flow) {
    for (int y = 0; y < flow.rows; y++) {
        for (int x = 0; x < flow.cols; x++) {

            // from + fwd = to
            Point from(x, y);
            const Point2f &fwd = flow.at<Point2f>(y, x);
            Point to(cvRound(x + fwd.x), cvRound(y + fwd.y));

            if (pointPolygonTest(prevContour, from, false) > 0.0 &&
                pointPolygonTest(currentContour, to, false) > 0.0) {

            }
        }
    }
}*/

cv::Point2f OpticalFlowCalculator::calcAverageMovement(const std::vector<cv::Point2f> &points1,
                                                       const std::vector<cv::Point2f> &points2) {

    Point2f sum(0.0, 0.0);

    for (int i = 0; i < points1.size(); i++) {
        sum += (points2[i] - points1[i]);
    }

    return sum / ((int) points1.size());
}
