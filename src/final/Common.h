#pragma once

#include <iostream>
#include <opencv2/core/types.hpp>
#include <unordered_map>
#include <map>
#include <opencv2/video/background_segm.hpp>
#include "camera/Camera.hpp"
#include "camera/CameraPose.h"
#include "SingleObjectSelector.hpp"

struct CloudPoint {
    cv::Point3d pt;

    //std::vector<int> imgpt_for_img;

    double reprojection_error;
};

std::pair<int, int> makePair(const cv::Point &pt);

struct Cloud {

    // the points included in this cloud
    std::vector<CloudPoint> points;

    std::unordered_map<long, std::map<std::pair<int, int>, int>> lookupIdxBy2D;

    // FIXME -- this could me map<long, vector>>
    std::unordered_map<long, std::map<int, cv::Point2i>> lookup2DByIdx;

    void insert(long frameId, CloudPoint cp, cv::Point2i pt2d) {
        points.push_back(cp);

        map(points.size() - 1, frameId, pt2d);
    }

    void map(unsigned long idx, long frameId, cv::Point2i pt2d) {
        lookupIdxBy2D[frameId][makePair(pt2d)] = idx;
        lookup2DByIdx[frameId][idx] = pt2d;
    }

    void insert(Cloud &other1, long frameId1, Cloud &other2, long frameId2, unsigned int idx) {
        cv::Point2i pt2d_1(other1.lookup2DByIdx[frameId1][idx]);
        cv::Point2i pt2d_2(other2.lookup2DByIdx[frameId2][idx]);

        insert(frameId1, other1.points[idx], pt2d_1);

        map(points.size() - 1, frameId2, pt2d_2);
    }

    void insert(CloudPoint cp, long frameId1, cv::Point2i pt1, long frameId2, cv::Point2i pt2) {
        points.push_back(cp);

        map(points.size() - 1, frameId1, pt1);
        map(points.size() - 1, frameId2, pt2);
    }

    size_t size() const {
        return points.size();
    }

    void clear() {
        points.clear();
        lookupIdxBy2D.clear();
        lookup2DByIdx.clear();
    }
};

std::vector<cv::Point3d> CloudPointsToPoints(const std::vector<CloudPoint> &cpts);

void writeCloudPoints(const std::vector<CloudPoint> &cpts);

void writeCloudPoints(const std::string &fileName, const std::vector<CloudPoint> &cpts);

void translate(const std::vector<cv::Point> &input, cv::Point translation, std::vector<cv::Point> &output);

void translate(std::vector<cv::Point> &input, cv::Point translation);

cv::Mat mergeImages(const cv::Mat &left, const cv::Mat &right);
cv::Mat mergeImagesVertically(const cv::Mat &left, const cv::Mat &right);

void shiftImage(const cv::Mat &input, const cv::Rect &boundingRect,
                const cv::Point2i &translation, cv::Mat &output);

bool CheckCoherentRotation(cv::Mat_<double> &R);

bool FindPoseEstimation(
        cv::Ptr<Camera> camera,
        cv::Mat_<double> &rvec,
        cv::Mat_<double> &t,
        cv::Mat_<double> &R,
        std::vector<cv::Point3f> ppcloud,
        std::vector<cv::Point2f> imgPoints,
        std::vector<cv::Point2f> &reprojected
);

void drawBoxOnChessboard(cv::Mat inputImage, cv::Ptr<Camera> camera, cv::Ptr<CameraPose> pose);

void drawGridXY(cv::Mat &img, cv::Ptr<Camera> camera, cv::Ptr<CameraPose> cameraPose);

cv::Point moveToTheCenter(cv::Mat image, cv::Mat mask);

std::vector<cv::Mat> getFramesFromCameras(std::vector<cv::Ptr<Camera>> &camera,
                                          std::vector<cv::Ptr<cv::BackgroundSubtractorMOG2>> &bgSub,
                                          std::vector<SingleObjectSelector> &objSelector,
                                          double learningRate);

cv::Point2f magicVector(const std::vector<cv::Point2f> &vector);

cv::Mat slerp(cv::Mat rvec1, cv::Mat rvec2, double ratio);
