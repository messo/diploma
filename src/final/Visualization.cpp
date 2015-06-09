#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <iomanip>
#include "Visualization.h"

using namespace cv;

void Visualization::renderWithDepth(const std::vector<CloudPoint> &points) {
    double t0 = getTickCount();

    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;

    // FIXME -- inefficient?
    std::vector<double> depths;
    for (int i = 0; i < points.size(); i++) {
        if (points[i].reprojection_error < REPROJ_ERROR_THRESHOLD) {
            objectPoints.push_back(points[i].pt);
            depths.push_back(points[i].pt.z);
        }
    }

    Mat img(SIZE, CV_8UC3, Scalar(0, 0, 0));

    if (RATIO != 1.0f) {
        img = Mat(Size(SIZE.width / RATIO, SIZE.height / RATIO), CV_8UC3, Scalar(0, 0, 0));
    }

//    const double maxZ = -74.0;
//    const double minZ = -88.0;

    double minZ, maxZ;
    minMaxLoc(depths, &minZ, &maxZ);
//    std::cout << minZ << " " << maxZ;

    if (objectPoints.size() > 0) {
        projectPoints(objectPoints, cameraPose.rvec, cameraPose.tvec, cameraMatrix, cv::noArray(), imagePoints);

        for (int i = 0; i < imagePoints.size(); i++) {
//            std::cout << objectPoints[i].z << std::endl;
            double _d = MAX(MIN((objectPoints[i].z - minZ) / (maxZ - minZ), 1.0), 0.0);
//            std::cout << _d << " " << ((int) (_d * 180)) << std::endl;
            circle(img, (RATIO == 1.0f) ? (imagePoints[i] - (LEFT_SHIFT + RIGHT_SHIFT) / 2) : (imagePoints[i] / RATIO), 1, Scalar((int) (_d * 160), 255, 255),
                   1);
        }
    }

//    for(int i=0; i<640; i++) {
//        double _d = ((double)i) / 640;
//        circle(img, Point2f(i, 450), 1, Scalar((int) (_d * 180), 255, 255), 1);
//    }

    ScopedLock lock(mutexType);
    cvtColor(img, result, COLOR_HSV2BGR);

    t0 = ((double) getTickCount() - t0) / getTickFrequency();
    std::cout << "[" << std::setw(20) << "Visualization" << "] " << "Done in " << t0 << "s" << std::endl;
    std::cout.flush();
}

void Visualization::renderWithColors(const std::vector<CloudPoint> &points,
                                     const std::vector<cv::Point2f> &originalPoints,
                                     const cv::Mat &image) {
    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;

    // FIXME -- inefficient?
    std::vector<Vec3b> colors;
    for (int i = 0; i < points.size(); i++) {
        if (points[i].reprojection_error < REPROJ_ERROR_THRESHOLD) {
            objectPoints.push_back(points[i].pt);
            colors.push_back(image.at<Vec3b>(Point2i(originalPoints[i])));
        }
    }

    Mat img(480, 640, CV_8UC3, Scalar(0, 0, 0));

    if (objectPoints.size() > 0) {
        projectPoints(objectPoints, cameraPose.rvec, cameraPose.tvec, cameraMatrix, cv::noArray(), imagePoints);

        for (int i = 0; i < imagePoints.size(); i++) {
            img.at<Vec3b>(Point2i(imagePoints[i])) = colors[i];
        }
    }

    ScopedLock lock(mutexType);
    img.copyTo(result);
}

void Visualization::renderWithGrayscale(const std::vector<CloudPoint> &points,
                                        const std::vector<cv::Point2f> &originalPoints,
                                        const cv::Mat &image) {
    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;

    // FIXME -- inefficient?
    std::vector<uchar> colors;
    for (int i = 0; i < points.size(); i++) {
        if (points[i].reprojection_error < REPROJ_ERROR_THRESHOLD) {
            objectPoints.push_back(points[i].pt);
            colors.push_back(image.at<uchar>(Point2i(originalPoints[i])));
        }
    }

    Mat img(480, 640, CV_8UC3, Scalar(0, 0, 0));

    if (objectPoints.size() > 0) {
        projectPoints(objectPoints, cameraPose.rvec, cameraPose.tvec, cameraMatrix, cv::noArray(), imagePoints);

        for (int i = 0; i < imagePoints.size(); i++) {
            img.at<Vec3b>(Point2i(imagePoints[i])) = Vec3b(colors[i], colors[i], colors[i]);
        }
    }

    ScopedLock lock(mutexType);
    img.copyTo(result);
}

void Visualization::renderWithContours(const std::vector<CloudPoint> &points) {
    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;

    for (int i = 0; i < points.size(); i++) {
        if (points[i].reprojection_error < REPROJ_ERROR_THRESHOLD) {
            objectPoints.push_back(points[i].pt);
        }
    }

    Mat img(480, 640, CV_8U, Scalar(0));

    if (objectPoints.size() > 0) {
        projectPoints(objectPoints, cameraPose.rvec, cameraPose.tvec, cameraMatrix, cv::noArray(), imagePoints);

        for (int i = 0; i < imagePoints.size(); i++) {
            circle(img, imagePoints[i], 2, Scalar(255), 1, LINE_AA);
        }
    }

    dilate(img, img, Mat(), Point(-1, -1), 2);
    erode(img, img, Mat(), Point(-1, -1), 2);

    std::vector<std::vector<Point>> contours;
    findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat contourImg(480, 640, CV_8U, Scalar(0));
    for (int i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > 70.0) {
            drawContours(contourImg, contours, i, Scalar(255), 2, LINE_AA);
        }
    }

    ScopedLock lock(mutexType);
    contourImg.copyTo(result);
}

void Visualization::renderWithContours(const std::vector<std::vector<CloudPoint>> &pppoints) {

    Mat contourImg(480, 640, CV_8U, Scalar(0));

    for (int j = 0; j < pppoints.size(); j++) {

        const std::vector<CloudPoint> &points = pppoints[j];

        std::vector<cv::Point3f> objectPoints;
        std::vector<cv::Point2f> imagePoints;

        for (int i = 0; i < points.size(); i++) {
            if (points[i].reprojection_error < REPROJ_ERROR_THRESHOLD) {
                objectPoints.push_back(points[i].pt);
            }
        }

        Mat img(480, 640, CV_8U, Scalar(0));

        if (objectPoints.size() > 0) {
            projectPoints(objectPoints, cameraPose.rvec, cameraPose.tvec, cameraMatrix, cv::noArray(), imagePoints);

            for (int i = 0; i < imagePoints.size(); i++) {
                circle(img, imagePoints[i], 2, Scalar(255), 1, LINE_AA);
            }
        }

        dilate(img, img, Mat(), Point(-1, -1), 2);
        erode(img, img, Mat(), Point(-1, -1), 2);

        std::vector<std::vector<Point>> contours;
        findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        for (int i = 0; i < contours.size(); i++) {
            double area = contourArea(contours[i]);
            if (area > 70.0) {
                drawContours(contourImg, contours, i, Scalar(255), 2, LINE_AA);
            }
        }
    }

    ScopedLock lock(mutexType);
    contourImg.copyTo(result);
}

cv::Mat Visualization::getResult() {
    ScopedLock lock(mutexType);

    return result.clone();
}
