#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui.hpp>
#include "../Common.h"
#include "ReportOpticalFlowCalculator.h"

using namespace cv;
using namespace std;


pair<vector<Point2f>, vector<Point2f>> ReportOpticalFlowCalculator::calcDenseMatches(std::vector<cv::Mat> &frames, const Object &object) {

    double t0 = getTickCount();

#pragma omp parallel for
    for (int i = 0; i < 2; i++) {
        if (frames[i].channels() != 1) {
            cvtColor(frames[i], this->frames[i], COLOR_BGR2GRAY);
        } else {
            this->frames[i] = frames[i].clone();
        }
        this->masks[i] = object.masks[i].clone();

//         this->texturedRegions[i] = masks[i].clone();
        this->texturedRegions[i] = this->calcTexturedRegion(this->frames[i], this->masks[i]);
    }

    // SHIFTING....
//    Point2i optimalShift = getOptimalShift();
//
//    shiftFrame(1, -optimalShift);

//    Rect unified = boundingRect(this->masks[0]) | boundingRect(this->masks[1]);
//    Mat merged = mergeImagesVertically(this->frames[0](unified), this->frames[1](unified));
//    imwrite("/media/balint/Data/Linux/diploma/after_shift.png", merged);

    imshow("frame1", this->frames[0]);
    imshow("frame2", this->frames[1]);

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
    std::cout << "Feed init done in " << t0 << "s" << std::endl;
    std::cout.flush();

    std::vector<Mat> flows = this->calcOpticalFlows();
    std::pair<std::vector<Point2f>, std::vector<Point2f>> matches = collectMatchingPoints(flows);

    visualizeMatchesROI(frames[0], matches.first, frames[1], matches.second);

    std::cout << matches.first.size() << std::endl;
    std::cout.flush();

//    this->visualizeMatches(points1, points2);

    // move the points to their original location
    for (int i = 0; i < matches.second.size(); i++) {
        //matches.second[i] += Point2f(optimalShift);
    }

    return matches;
}

std::vector<cv::Mat> ReportOpticalFlowCalculator::calcOpticalFlows() const {
    std::vector<cv::Mat> flows(2);
    Mat _flows[2];

    // find a common bounding rect to simplify the OF
    Rect rect0(boundingRect(masks[0]));
    Rect rect1(boundingRect(masks[1]));
    Rect unified(rect0 | rect1);

#pragma omp parallel for
    for (int i = 0; i < 2; i++) {
        //std::cout << "THREAD: " << omp_get_thread_num() << std::endl;
        calcOpticalFlowFarneback(frames[i](unified), frames[(i + 1) % 2](unified), _flows[i], 0.75, 6, 21, 10, 7, 1.5,
                                 OPTFLOW_FARNEBACK_GAUSSIAN);

        // put the flows back into the full matrix
        flows[i] = Mat::zeros(frames[0].rows, frames[0].cols, CV_32FC2);
        _flows[i].copyTo(flows[i](unified));
    }

//    this->visualizeOpticalFlow(frames[0], masks[0], frames[1], masks[1], flows[0], "flow12");
//    this->visualizeOpticalFlow(frames[1], masks[1], frames[0], masks[0], flows[1], "flow21");

    // trying to smooth the vectorfield
    //Mat smoothedFlow;
    //GaussianBlur(flow2, smoothedFlow, Size(15, 15), -1);
    //smoothedFlow.copyTo(flow2);

    // collecting matching points using optical flow

    return flows;
}

std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> ReportOpticalFlowCalculator::collectMatchingPoints(const std::vector<cv::Mat> &flows) const {

    Rect rect0(boundingRect(masks[0]));
    Rect rect1(boundingRect(masks[1]));
    Rect roi(rect0 | rect1);

    std::vector<Point2f> points1, points2;

//    double t0 = getTickCount();

    Rect imageArea(Point(0, 0), masks[1].size());

    const Mat &flow = flows[0];
    const Mat &backFlow = flows[1];

    for (int y = roi.tl().y; y <= roi.br().y; y++) {
        for (int x = roi.tl().x; x <= roi.br().x; x++) {

            // from + fwd = to
            Point from(x, y);
            const Point2f &fwd = flow.at<Point2f>(from);
            Point to(cvRound(x + fwd.x), cvRound(y + fwd.y));

            // check if we are still in the mask, and the movement is not zero
            if ((masks[0].at<uchar>(from) == 255) && to.inside(imageArea) && (masks[1].at<uchar>(to) == 255)) {

                // to + back = backTo
                const Point2f &back = backFlow.at<Point2f>(to);
                Point backTo(cvRound(to.x + back.x), cvRound(to.y + back.y));

                Point diff2(backTo.x - from.x, backTo.y - from.y);

                // only use if the backflow points almost there
                if (diff2.dot(diff2) <= 1) {
                    if (texturedRegions[0].at<char>(from) && texturedRegions[1].at<char>(to)) {
                        points1.push_back(Point2f(from));
                        points2.push_back(Point2f(x + fwd.x, y + fwd.y));
                    }
                }
            }
        }
    }

    return make_pair(points1, points2);
}
