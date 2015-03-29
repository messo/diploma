#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "ObjectSelector.hpp"

using namespace std;
using namespace cv;

ObjectSelector::ObjectSelector() {

}

cv::Mat ObjectSelector::selectUsingConnectedComponents(const cv::Mat &img, const cv::Mat &mask) {
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

cv::Mat ObjectSelector::selectUsingContoursWithMaxArea(const cv::Mat &img, const cv::Mat &mask) {
    vector<vector<Point> > contours;

    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat dst = Mat::zeros(mask.size(), CV_8UC3);

    if (contours.size() == 0)
        return dst;

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

    Scalar color(255, 255, 255);
    drawContours(dst, contours, largestComp, color, FILLED, LINE_8);

    Mat result;
    img.copyTo(result, dst);
    return result;
}

cv::Mat ObjectSelector::selectUsingContoursWithClosestCentroid(const cv::Mat &img, const cv::Mat &mask) {
    vector<vector<Point> > contours;

    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    lastMask = Mat::zeros(mask.size(), CV_8U);

    if (contours.size() == 0)
        return lastMask;

    int selectedComponent = 0;

    if (lastCentroid.empty()) {
        double maxArea = 0;

        for (int idx = 0; idx < contours.size(); idx++) {
            const vector<Point> &c = contours[idx];
            double area = fabs(contourArea(Mat(c)));
            if (area > maxArea) {
                maxArea = area;
                selectedComponent = idx;
            }
        }

    } else {
        double minDistance = DBL_MAX;

        for (int idx = 0; idx < contours.size(); idx++) {
            Ptr<Point> currentCentroid = getCentroid(moments(contours[idx]));

            Point diff(currentCentroid.get()->x - lastCentroid.get()->x, currentCentroid.get()->y - lastCentroid.get()->y);
            double distance = diff.dot(diff);

            if (distance < minDistance) {
                minDistance = distance;
                selectedComponent = idx;
            }
        }
    }

    lastCentroid = getCentroid(moments(contours[selectedComponent]));

    drawContours(lastMask, contours, selectedComponent, Scalar(255), FILLED, LINE_8);

    // save the contour
    lastContour = contours[selectedComponent];
    lastBoundingRect = boundingRect(lastContour);

    Mat result;
    img.copyTo(result, lastMask);
    return result;
}

cv::Ptr<cv::Point> ObjectSelector::getCentroid(cv::Moments moments) {
    int x = static_cast<int>(moments.m10 / moments.m00);
    int y = static_cast<int>(moments.m01 / moments.m00);

    return cv::Ptr<cv::Point>(new Point(x, y));
}

const std::vector<cv::Point> &ObjectSelector::getLastContour() const {
    return lastContour;
}
