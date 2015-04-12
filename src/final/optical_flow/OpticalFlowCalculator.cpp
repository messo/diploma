#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui.hpp>
#include "OpticalFlowCalculator.h"
#include "../Common.h"
#include "OFReconstruction.h"

using namespace cv;
using namespace std;

void OpticalFlowCalculator::calcTexturedRegions(const Mat frame, const Mat mask, Mat &texturedRegions) const {
    // H(σ(Iy - Ix) - ε)
    int ksize = 1;

    texturedRegions.create(frame.rows, frame.cols, CV_8U);
    texturedRegions.setTo(0);

    // FIXME -- this can be done in boundringRect!

    for (int y = 0; y < frame.rows; y++) {
        for (int x = 0; x < frame.cols; x++) {
            if (!frame.at<char>(y, x)) {
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

double OpticalFlowCalculator::calcOpticalFlow(Point &translation) {

    Mat flow12;
    calcOpticalFlowFarneback(frame1, frame2, flow12, 0.5, 4, 21, 10, 7, 1.5, OPTFLOW_FARNEBACK_GAUSSIAN);

//    this->visualizeOpticalFlow(frame1, mask1, frame2, mask2, flow12, "flow12");

    Mat flow21;
    calcOpticalFlowFarneback(frame2, frame1, flow21, 0.5, 4, 21, 10, 7, 1.5, OPTFLOW_FARNEBACK_GAUSSIAN);

//    this->visualizeOpticalFlow(frame2, mask2, frame1, mask1, flow21, "flow21");

    // trying to smooth the vectorfield
    //Mat smoothedFlow;
    //GaussianBlur(flow2, smoothedFlow, Size(15, 15), -1);
    //smoothedFlow.copyTo(flow2);

    // collecting matching points using optical flow
    collectMatchingPoints(flow12, flow21, points1, points2);

//    visualizeMatches(points1, points2);
//    waitKey();

    Point2f avgMovement(this->calcAverageMovement(points1, points2));

    double length = norm(avgMovement);
    cout << avgMovement << ": " << length << endl;
    cout.flush();

    if (length > 10.0) {
        // we consider this as "too big" displacement, so we shift the current image further, and do this again

        Point newTranslation(-avgMovement);

        Rect currentBoundingRect(boundingRect(mask2));

        // translate the image
        Mat _currentFrame;
        frame2.copyTo(_currentFrame);
        shiftImage(_currentFrame, currentBoundingRect, newTranslation, frame2);
        // translate the mask
        Mat _currentMask;
        mask2.copyTo(_currentMask);
        shiftImage(_currentMask, currentBoundingRect, newTranslation, mask2);
        // translate the textured regions
        Mat _currentTexturedRegions;
        texturedRegions2.copyTo(_currentTexturedRegions);
        shiftImage(_currentTexturedRegions, currentBoundingRect, newTranslation, texturedRegions2);
        // translating the contour
        // translate(currentContour, translation);

        translation += newTranslation;

        length = this->calcOpticalFlow(translation);
    } else { //if (length >= 1.0) {*/
//        this->visualizeMatches(points1, points2);

        // reconstruction
//        currentReconstruction = Ptr<OFReconstruction>(new OFReconstruction(camera, points1, points2));
//        currentReconstruction->reconstruct();
        //} else {
//        imshow("prev", prevFrame);
//        imshow("this", currentFrame);
    }

    return 2.0; //length;
}

void OpticalFlowCalculator::collectMatchingPoints(const Mat &flow, const Mat &backFlow,
                                                  vector<Point2f> &points1, vector<Point2f> &points2) {
    points1.clear();
    points2.clear();

    for (int y = 0; y < flow.rows; y++) {
        for (int x = 0; x < flow.cols; x++) {

            // from + fwd = to
            Point from(x, y);
            const Point2f &fwd = flow.at<Point2f>(from);
            Point to(cvRound(x + fwd.x), cvRound(y + fwd.y));

            // check if we are still in the mask, and the movement is not zero
            if (mask1.at<uchar>(from) && mask2.at<uchar>(to) && fwd.dot(fwd) > 0.0001) {

                // to + back = backTo
                const Point2f &back = backFlow.at<Point2f>(to);
                Point backTo(cvRound(to.x + back.x), cvRound(to.y + back.y));

                Point diff2(backTo.x - from.x, backTo.y - from.y);

                // only use if the backflow points almost there
                if (diff2.dot(diff2) <= 1) {
                    if (texturedRegions1.at<char>(from) && texturedRegions2.at<char>(to)) {
                        points1.push_back(Point2f(from));
                        points2.push_back(Point2f(x + fwd.x, y + fwd.y));
                    }
                }

            }
        }
    }
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
    frame2.copyTo(vis);
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
    if(vis.channels() != 3) {
        cvtColor(vis, vis, cv::COLOR_GRAY2BGR);
    }

    RNG rng;
    for (int i = 0; i < points1.size(); i++) {
        Point p1(points1[i]);
        Point p2(points2[i]);

        if (p1.x % 10 == 0 && p1.y % 10 == 0) {
            int icolor = (unsigned) rng;
            Scalar color(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);

            line(vis, p1, p2 + Point(640, 0), color);
            circle(vis, p1, 2, color, -1);
            circle(vis, p2 + Point(640, 0), 2, color, -1);
        }
    }

    imshow("restoredMatchesVis", vis);
}

cv::Point2f OpticalFlowCalculator::calcAverageMovement(const std::vector<cv::Point2f> &points1,
                                                       const std::vector<cv::Point2f> &points2) const {

    Point2f sum(0.0, 0.0);

    for (int i = 0; i < points1.size(); i++) {
        sum += (points2[i] - points1[i]);
    }

    return sum / ((int) points1.size());
}