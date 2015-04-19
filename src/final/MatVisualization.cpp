#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include "MatVisualization.h"

using namespace cv;

void MatVisualization::renderPointCloud(const std::vector<CloudPoint> &points) {
    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;

    // FIXME -- inefficient?
    for (int i = 0; i < points.size(); i++) {
        if (points[i].reprojection_error < 5.0) {
            objectPoints.push_back(points[i].pt);
        }
    }

    Mat img(480, 640, CV_8UC3, Scalar(0, 0, 0));

    const double maxZ = 10.0;
    const double minZ = -50.0;

    if (objectPoints.size() > 0) {
        projectPoints(objectPoints, cameraPose->rvec, cameraPose->tvec, cameraMatrix, cv::noArray(), imagePoints);

        for (int i = 0; i < imagePoints.size(); i++) {
            double _d = MAX(MIN((objectPoints[i].z - minZ) / (maxZ - minZ), 1.0), 0.0);
            circle(img, imagePoints[i], 1, Scalar((int) (_d * 255), 255, 255), 1);
        }
    }

    ScopedLock lock(mutexType);
    cvtColor(img, result, COLOR_HSV2BGR);
}

cv::Mat MatVisualization::getResult() {
    ScopedLock lock(mutexType);

    return result.clone();
}
