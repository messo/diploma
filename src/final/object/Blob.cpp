#include <opencv2/imgproc.hpp>
#include "Blob.h"

using namespace cv;

Blob::Blob(std::vector<Point> &contour) : contour(contour), mask(480, 640, CV_8U, Scalar(0)) {

    std::vector<std::vector<Point>> contours(1);
    contours[0] = contour;

    cv::drawContours(mask, contours, 0, Scalar(255), -1, LINE_AA);

}

bool Blob::contains(const cv::Point2f &point) const {
    return mask.at<uchar>(Point2i(point));
}
