#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "SingleObjectSelector.hpp"
#include "../PerformanceMonitor.h"

using namespace std;
using namespace cv;


std::vector<Object> SingleObjectSelector::selectObjects(const std::vector<cv::Mat> &frames,
                                                        const std::vector<cv::Mat> &masks) {

    PerformanceMonitor::get()->objSelectionStarted();
    std::vector<Mat> newMasks(2);

//    for (int i = 0; i < 2; i++) {
//        Mat labels, stats, centroids;
//        int nLabels = connectedComponentsWithStats(masks[i], labels, stats, centroids);
//
//        // select biggest component
//        int maxArea = 0;
//        int selectedLabel = -1;
//        for (int j = 1; j < nLabels; j++) {
//            int a = stats.at<int>(j, CC_STAT_AREA);
//            if (a > maxArea) {
//                maxArea = a;
//                selectedLabel = j;
//            }
//        }
//
//        newMasks[i] = Mat::zeros(frames[i].rows, frames[i].cols, CV_8U);
//
//        for (int y = 0; y < labels.rows; y++) {
//            for (int x = 0; x < labels.cols; x++) {
//                if (labels.at<int>(y, x) == selectedLabel) {
//                    newMasks[i].at<uchar>(y, x) = 255;
//                }
//            }
//        }
//    }



    for (int i = 0; i < 2; i++) {
        std::vector<std::vector<Point>> allContours;

        Mat img;
        masks[i].copyTo(img);
        findContours(img, allContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        double maxArea = 380.0;
        int selectedContour = -1;
        for (int idx = 0; idx < allContours.size(); idx++) {
            const std::vector<Point> &c = allContours[idx];
            double area = fabs(contourArea(c));
            if (area > maxArea) {
                maxArea = area;
                selectedContour = idx;
            }
        }

        if(selectedContour != -1) {
            newMasks[i] = Mat::zeros(frames[i].rows, frames[i].cols, CV_8U);
            cv::drawContours(newMasks[i], allContours, selectedContour, Scalar(255), -1, LINE_8);
        }

//        for(int j=0; j<allContours.size(); j++) {
//            cv::drawContours(newMasks[i], allContours, j, j==selectedContour ? Scalar(255) : Scalar(100), -1, LINE_AA);
//        }
    }

    if(newMasks[0].empty() || newMasks[1].empty()) {
        return std::vector<Object>();
    }

    vector<pair<Point2f, Point2f>> matches = matcher.match(frames, newMasks);

    std::vector<Object> result;
    result.push_back(Object(newMasks[0], newMasks[1]));
    result.back().matches = matches;

    PerformanceMonitor::get()->objSelectionFinished();

    return result;
}
