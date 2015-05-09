#include "OpticalFlowCalculator.h"

#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui.hpp>
#include <iomanip>
#include <opencv2/cudaoptflow.hpp>
#include "../Common.h"
#include "../PerformanceMonitor.h"

using namespace cv;
using namespace cv::cuda;
using namespace std;

pair<vector<Point2f>, vector<Point2f>> OpticalFlowCalculator::calcDenseMatches(std::vector<cv::Mat> &frames, const Object &object) {

    PerformanceMonitor::get()->ofInitStarted();

    double t0 = getTickCount();

#pragma omp parallel for
    for (int i = 0; i < 2; i++) {
        if (frames[i].channels() != 1) {
            cvtColor(frames[i], this->frames[i], COLOR_BGR2GRAY);
        } else {
            this->frames[i] = frames[i].clone();
        }
        this->masks[i] = object.masks[i]; //.clone();

//         this->texturedRegions[i] = masks[i].clone();
        this->texturedRegions[i] = this->calcTexturedRegion(this->frames[i], this->masks[i]);
    }

    // SHIFT
    Point optimalShift;
    if (object.matches.size() >= 2) {
        Point2f sum;
        for (int i = 0; i < object.matches.size(); i++) {
            sum += (object.matches[i].second - object.matches[i].first);
        }
        optimalShift = Point2i(sum / ((int) object.matches.size()));

        std::cout << "[" << std::setw(20) << "OFCalculator" << "] " << "SHIFT: " << optimalShift << endl;
        shiftFrame(1, -optimalShift);
    } else {
        std::cerr << "[" << std::setw(20) << "OFCalculator" << "] " << "Not enough matches, no shifting!" << endl;
    }

//    Rect unified = boundingRect(this->masks[0]) | boundingRect(this->masks[1]);
//    Mat merged = mergeImagesVertically(this->frames[0](unified), this->frames[1](unified));
//    imwrite("/media/balint/Data/Linux/diploma/after_shift.png", merged);

//    imshow("frame1", this->frames[0]);
//    imshow("frame2", this->frames[1]);

    // --------------------

//    Rect br0 = boundingRect(this->masks[0]);
//    Rect br1 = boundingRect(this->masks[1]);
//    Rect uni = br0 | br1;
//    Mat frame0 = frames[0].clone();
//    Mat frame1 = frames[1].clone();
//    rectangle(frame0, uni.tl(), uni.br(), Scalar(0, 0, 255), 2, LINE_AA);
//    rectangle(frame0, br0.tl(), br0.br(), Scalar(0, 255, 255), 1, LINE_AA);
//    rectangle(frame1, uni.tl(), uni.br(), Scalar(0, 0, 255), 2, LINE_AA);
//    rectangle(frame1, br1.tl(), br1.br(), Scalar(0, 255, 255), 1, LINE_AA);
//    imwrite("/media/balint/Data/Linux/diploma/of_img_left_framed.png", frame0);
//    imwrite("/media/balint/Data/Linux/diploma/of_img_right_framed.png", frame1);

    // texturazott cuccok...
//    Rect br0 = boundingRect(this->masks[0]);
//    Rect br1 = boundingRect(this->masks[1]);
//    Mat textures = mergeImages(this->texturedRegions[0](br0), this->texturedRegions[1](br1));
//    imshow("textures", textures);
//    imwrite("/media/balint/Data/Linux/diploma/textures.png", textures);


    t0 = ((double) getTickCount() - t0) / getTickFrequency();
    std::cout << "[" << std::setw(20) << "OFCalculator" << "] " << "Feed init done in " << t0 << "s" << std::endl;
    std::cout.flush();
    PerformanceMonitor::get()->ofInitFinished();

    std::vector<Mat> flows = this->calcOpticalFlows();

    pair<vector<Point2f>, vector<Point2f>> matches = collectMatchingPoints(flows);

//    this->visualizeMatches(points1, points2);

    // move the points to their original location
    for (int i = 0; i < matches.second.size(); i++) {
        matches.second[i] += Point2f(optimalShift);
    }

//    this->visualizeMatches(frames[0], points1, frames[1], points2);

    return matches;
}

Mat OpticalFlowCalculator::calcTexturedRegion(const Mat frame, const Mat mask) const {
    // H(σ(Iy - Ix) - ε)
    int ksize = 1;

    Mat texturedRegion(frame.rows, frame.cols, CV_8U, Scalar(0));

    Rect roi(boundingRect(mask));

//#pragma omp parallel for collapse(2)
    for (int y = roi.tl().y; y <= roi.br().y; y++) {
        for (int x = roi.tl().x; x <= roi.br().x; x++) {
            if (!mask.at<char>(y, x)) {
                continue;
            }

            uchar Ixy = frame.at<uchar>(y, x);

            vector<int> values(9);
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
                texturedRegion.at<uchar>(y, x) = 255;
            }
        }
    }

    return texturedRegion;
}

std::vector<Mat> OpticalFlowCalculator::calcOpticalFlows() const {

    PerformanceMonitor::get()->ofCalcStarted();
    double t0 = getTickCount();

    std::vector<Mat> flows(2);
    Mat _flows[2];

    // find a common bounding rect to simplify the OF
    Rect rect0(boundingRect(masks[0]));
    Rect rect1(boundingRect(masks[1]));
    Rect unified(rect0 | rect1);

#pragma omp parallel for
    for (int i = 0; i < 2; i++) {

        Ptr<FarnebackOpticalFlow> of = FarnebackOpticalFlow::create(6, 0.75, false, 21, 10, 7, 1.5, OPTFLOW_FARNEBACK_GAUSSIAN);

        GpuMat d_frameL(frames[i](unified)), d_frameR(frames[(i + 1) % 2](unified));
        GpuMat d_flow;

        of->calc(d_frameL, d_frameR, d_flow, Stream::Null());

        Mat flow;
        d_flow.download(flow);

//        this->visualizeOpticalFlow(frame1, mask1, frame2, mask2, flow12, "flow12");

        // put the flows back into the full matrix
        flows[i] = Mat::zeros(frames[0].rows, frames[0].cols, CV_32FC2);
        flow.copyTo(flows[i](unified));
    }

//    this->visualizeOpticalFlow(frames[0], masks[0], frames[1], masks[1], flows[0], "flow12");
//    this->visualizeOpticalFlow(frames[1], masks[1], frames[0], masks[0], flows[1], "flow21");

    // collecting matching points using optical flow

    t0 = ((double) getTickCount() - t0) / getTickFrequency();
    std::cout << "[" << std::setw(20) << "OFCalculator" << "] Optical flows have been calculated in " << t0 << "s" << std::endl;
    std::cout.flush();
    PerformanceMonitor::get()->ofCalcFinished();

//    visualizeMatchesROI(frames[0], points1, frames[1], points2);

    return flows;
}

pair<vector<Point2f>, vector<Point2f>> OpticalFlowCalculator::collectMatchingPoints(const vector<Mat> &flows) const {
    double t0 = getTickCount();
    PerformanceMonitor::get()->ofMatchingStarted();

    Rect rect0(boundingRect(masks[0]));
    Rect rect1(boundingRect(masks[1]));
    Rect roi(rect0 | rect1);

    vector<Point2f> points1, points2;

    Rect imageArea(Point(0, 0), masks[1].size());

    const Mat &flow = flows[0];
    const Mat &backFlow = flows[1];

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

    t0 = ((double) getTickCount() - t0) / getTickFrequency();
    std::cout << "[" << std::setw(20) << "OFCalculator" << "] Found matches: " << points1.size() << " in " << t0 << "s" << std::endl;
    std::cout.flush();
    PerformanceMonitor::get()->ofMatchingFinished();

    return make_pair(points1, points2);
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

                line(vis, from, to + Point(img1.cols, 0), color);
                circle(vis, from, 2, color, -1);
                circle(vis, to + Point(img1.cols, 0), 2, color, -1);
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

            line(vis, p1, p2 + Point(img1.cols, 0), color);
            circle(vis, p1, 2, color, -1);
            circle(vis, p2 + Point(img1.cols, 0), 2, color, -1);
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