#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "SingleObjectSelector.hpp"

using namespace std;
using namespace cv;


cv::Mat SingleObjectSelector::selectUsingConnectedComponents(const cv::Mat &img, const cv::Mat &mask) {
    Mat labels, stats, centroids;

    Mat selection(mask.size(), CV_8UC3);

    connectedComponentsWithStats(mask, labels, stats, centroids);

    for (int r = 0; r < selection.rows; ++r) {
        for (int c = 0; c < selection.cols; ++c) {
            int label = labels.at<int>(r, c);
            Vec3b &pixel = selection.at<Vec3b>(r, c);
            pixel = Vec3b(((label * 100) & 255), ((label * 100) & 255), ((label * 100) & 255));
        }
    }

    return selection;
}

cv::Mat SingleObjectSelector::selectUsingContourWithMaxArea(const cv::Mat &img, cv::Mat mask) {
    vector<vector<Point>> contours;

    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    lastMask = Mat::zeros(mask.size(), CV_8U);

    if (contours.size() != 0) {
        // iterate through all the top-level contours,
        // draw each connected component with its own random color
        int largestComp = 0;
        double maxArea = 0;

        for (int idx = 0; idx < contours.size(); idx++) {
            const vector<Point> &c = contours[idx];
            double area = fabs(contourArea(Mat(c)));
            if (area > maxArea) {
                maxArea = area;
                largestComp = idx;
            }
        }

        drawContours(lastMask, contours, largestComp, Scalar(255), FILLED, LINE_8);
    }

    Mat result;
    img.copyTo(result, lastMask);
    return result;
}

std::vector<Object> SingleObjectSelector::selectObjects(const std::vector<cv::Mat> &frames,
                                                        const std::vector<cv::Mat> &masks) {

    std::vector<Mat> newMasks(2);

    for (int i = 0; i < 2; i++) {
        Mat labels, stats, centroids;
        int nLabels = connectedComponentsWithStats(masks[i], labels, stats, centroids);

        // select biggest component
        int maxArea = 0;
        int selectedLabel = -1;
        for (int j = 1; j < nLabels; j++) {
            int a = stats.at<int>(j, CC_STAT_AREA);
            if (a > maxArea) {
                maxArea = a;
                selectedLabel = j;
            }
        }

        newMasks[i] = Mat::zeros(frames[i].rows, frames[i].cols, CV_8U);

        for (int y = 0; y < labels.rows; y++) {
            for (int x = 0; x < labels.cols; x++) {
                if (labels.at<int>(y, x) == selectedLabel) {
                    newMasks[i].at<uchar>(y, x) = 255;
                }
            }
        }
    }

    vector<pair<Point2f, Point2f>> matches = matcher.match(frames, newMasks);

    std::vector<Object> result;
    result.push_back(Object(newMasks[0], newMasks[1]));
    result.back().matches = matches;

    return result;
}
