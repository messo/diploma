#include "ReportOpticalFlowCalculator.h"
#include "SpatialOpticalFlowCalculator.h"
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui.hpp>
#include <omp.h>
#include "OpticalFlowCalculator.h"
#include "../Common.h"

using namespace cv;
using namespace std;

void OpticalFlowCalculator::calcTexturedRegions(const Mat frame, const Mat mask, Mat &texturedRegions) const {
    // H(σ(Iy - Ix) - ε)
    int ksize = 1;

    texturedRegions.create(frame.rows, frame.cols, CV_8U);
    texturedRegions.setTo(0);

    Rect roi(boundingRect(mask));

#pragma omp parallel for collapse(2)
    for (int y = roi.tl().y; y <= roi.br().y; y++) {
        for (int x = roi.tl().x; x <= roi.br().x; x++) {
            if (!mask.at<char>(y, x)) {
                continue;
            }

            uchar Ixy = frame.at<uchar>(y, x);

            vector<int> values;
            for (int j = y - ksize; j <= y + ksize; j++) {
                for (int i = x - ksize; i <= x + ksize; i++) {
                    if ((i == x && j == y) || i < 0 || j < 0 || i >= frame.cols || j >= frame.rows)
                        continue;

                    if (mask.at<uchar>(j, i)) {
                        values.push_back(((int) frame.at<uchar>(j, i)) - Ixy);
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
                texturedRegions.at<uchar>(y, x) = 255;
            }
        }
    }
}

double OpticalFlowCalculator::calcOpticalFlow() {

    Mat flows[2];
    Mat _flows[2];

    // find a common bounding rect to simplify the OF
    Rect rect0(boundingRect(masks[0]));
    Rect rect1(boundingRect(masks[1]));
    Rect unified(rect0 | rect1);

#pragma omp parallel for
    for (int i = 0; i < 2; i++) {
        calcOpticalFlowFarneback(frames[i](unified), frames[(i + 1) % 2](unified), _flows[i], 0.75, 6, 21, 10, 7, 1.5,
                                 OPTFLOW_FARNEBACK_GAUSSIAN);
//        this->visualizeOpticalFlow(frame1, mask1, frame2, mask2, flow12, "flow12");

        // put the flows back into the full matrix
        flows[i] = Mat::zeros(frames[0].rows, frames[0].cols, CV_32FC2);
        _flows[i].copyTo(flows[i](unified));
    }

//    this->visualizeOpticalFlow(frames[0], masks[0], frames[1], masks[1], flows[0], "flow12");
//    this->visualizeOpticalFlow(frames[1], masks[1], frames[0], masks[0], flows[1], "flow21");

    // collecting matching points using optical flow
    collectMatchingPoints(flows[0], flows[1], unified, points1, points2);

    std::cout << points1.size() << std::endl;
    std::cout.flush();

    Point2f avgMovement(this->calcAverageMovement(points1, points2));

    double length = norm(avgMovement);
    cout << avgMovement << ": " << length << endl;
    cout.flush();

    visualizeMatchesROI(frames[0], points1, frames[1], points2);

    return length;
}

void OpticalFlowCalculator::collectMatchingPoints(const Mat &flow, const Mat &backFlow, const Rect &roi,
                                                  vector<Point2f> &points1, vector<Point2f> &points2) {
    points1.clear();
    points2.clear();

//    double t0 = getTickCount();

    Rect imageArea(Point(0, 0), masks[1].size());

#pragma omp parallel for collapse(2)
    for (int y = roi.tl().y; y <= roi.br().y; y++) {
        for (int x = roi.tl().x; x <= roi.br().x; x++) {

            // from + fwd = to
            Point from(x, y);
            const Point2f &fwd = flow.at<Point2f>(from);
            Point to(cvRound(x + fwd.x), cvRound(y + fwd.y));

            // check if we are still in the mask, and the movement is not zero
            if (masks[0].at<uchar>(from) == 255 && to.inside(imageArea) && masks[1].at<uchar>(to) == 255) {

                // to + back = backTo
                const Point2f &back = backFlow.at<Point2f>(to);
                Point backTo(cvRound(to.x + back.x), cvRound(to.y + back.y));

                Point diff2(backTo.x - from.x, backTo.y - from.y);

                // only use if the backflow points almost there
                if (diff2.dot(diff2) <= 1) {
                    if (texturedRegions[0].at<char>(from) && texturedRegions[1].at<char>(to)) {
#pragma omp critical
                        {
                            points1.push_back(Point2f(from));
                            points2.push_back(Point2f(x + fwd.x, y + fwd.y));
                        }
                    }
                }

            }
        }
    }

//    t0 = ((double) getTickCount() - t0) / getTickFrequency();
//    std::cout << "collect Done in " << t0 << "s" << std::endl;
//    std::cout.flush();
}

// -----------------
//  VISUALIZATIONS
// -----------------

void OpticalFlowCalculator::visualizeOpticalFlow(const cv::Mat &img1, const cv::Mat &mask1, const cv::Mat &img2,
                                                 const cv::Mat &mask2, const cv::Mat &flow,
                                                 const std::string &name) const {
    Mat vis = mergeImages(img1, img2);
    cvtColor(vis, vis, cv::COLOR_GRAY2BGR);

    RNG rng;
    for (int y = 0; y < flow.rows; y += 30) {
        for (int x = 0; x < flow.cols; x += 30) {

            Point from(x, y);
            const Point2f &f = flow.at<Point2f>(y, x);
            Point to(cvRound(x + f.x), cvRound(y + f.y));

            if (mask1.at<uchar>(from) && mask2.at<uchar>(to)) {
                int icolor = (unsigned) rng;
                Scalar color(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);

                line(vis, from, to + Point(640, 0), color);
                circle(vis, from, 2, color, -1);
                circle(vis, to + Point(640, 0), 2, color, -1);
            }
        }
    }

    imshow(name, vis);
}

void OpticalFlowCalculator::visualizeMatches(const vector<Point2f> &points1, const vector<Point2f> &points2) const {
    Mat vis;
    frames[1].copyTo(vis);
    cvtColor(vis, vis, cv::COLOR_GRAY2BGR);

    for (int i = 0; i < points1.size(); i++) {
        Point p1(points1[i]);
        Point p2(points2[i]);

        if (p1.x % 10 == 0 && p1.y % 10 == 0) {
            line(vis, p1, p2, Scalar(0, 255, 0));
            circle(vis, p1, 2, Scalar(255, 0, 0), -1);
        }
    }

    imshow("plainMatchesVis", vis);
}

void OpticalFlowCalculator::visualizeMatches(const Mat &img1, const vector<Point2f> &points1,
                                             const Mat &img2, const vector<Point2f> &points2) const {

    Mat vis = mergeImages(img1, img2);
    if (vis.channels() != 3) {
        cvtColor(vis, vis, cv::COLOR_GRAY2BGR);
    }

    RNG rng;
    for (int i = 0; i < points1.size(); i++) {
        Point p1(points1[i]);
        Point p2(points2[i]);

        if (p1.x % 20 == 0 && p1.y % 20 == 0) {
            int icolor = (unsigned) rng;
            Scalar color(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);

            line(vis, p1, p2 + Point(640, 0), color);
            circle(vis, p1, 2, color, -1);
            circle(vis, p2 + Point(640, 0), 2, color, -1);
        }
    }

    imshow("restoredMatchesVis", vis);
}


void OpticalFlowCalculator::visualizeMatchesROI(cv::Mat const &img1, std::vector<cv::Point2f> const &points1,
                                                cv::Mat const &img2, std::vector<cv::Point2f> const &points2) {
    Rect br0 = boundingRect(masks[0]);
    Rect br1 = boundingRect(masks[1]);

    Mat vis = mergeImages(img1(br0), img2(br1));
    if (vis.channels() != 3) {
        cvtColor(vis, vis, cv::COLOR_GRAY2BGR);
    }

    RNG rng;
    for (int i = 0; i < points1.size(); i++) {
        Point p1(points1[i]);
        Point p2(points2[i]);

        if (p1.x % 20 == 0 && p1.y % 20 == 0) {
            int icolor = (unsigned) rng;
            Scalar color(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);

            line(vis, p1 - br0.tl(), p2 - br1.tl() + Point(br0.width, 0), color, 1, LINE_AA);
            circle(vis, p1 - br0.tl(), 3, color, -1);
            circle(vis, p2 - br1.tl() + Point(br0.width, 0), 3, color, -1);
        }
    }

    imshow("restoredMatchesVis", vis);
//    imwrite("/media/balint/Data/Linux/diploma/vis_full.png", vis);
}


cv::Point2f OpticalFlowCalculator::calcAverageMovement(const std::vector<cv::Point2f> &points1,
                                                       const std::vector<cv::Point2f> &points2) const {

    Point2f sum(0.0, 0.0);

    for (int i = 0; i < points1.size(); i++) {
        sum += (points2[i] - points1[i]);
    }

    return sum / ((int) points1.size());
}

void OpticalFlowCalculator::shiftFrame(int i, Point shift) {
    Rect currentBoundingRect(boundingRect(this->masks[i]));
    Mat _currentFrame;
    this->frames[i].copyTo(_currentFrame);
    shiftImage(_currentFrame, currentBoundingRect, shift, this->frames[i]);
    // translate the mask
    Mat _currentMask;
    this->masks[i].copyTo(_currentMask);
    shiftImage(_currentMask, currentBoundingRect, shift, this->masks[i]);
    // translate the textured regions
    Mat _currentTexturedRegions;
    texturedRegions[i].copyTo(_currentTexturedRegions);
    shiftImage(_currentTexturedRegions, currentBoundingRect, shift, texturedRegions[i]);

}