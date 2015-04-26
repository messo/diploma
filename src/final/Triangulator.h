/*
 *  Triangulation.h
 *  SfMToyLibrary
 *
 *  Created by Roy Shilkrot on 12/23/11.
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2013 Roy Shilkrot
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 *
 */

#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include "Common.h"

#define EPSILON 0.0001

class Triangulator {

    cv::Ptr<Camera> camera1;
    cv::Ptr<Camera> camera2;

    const CameraPose &pose1;
    const CameraPose &pose2;

public:

    Triangulator(const cv::Ptr<Camera> &cam1, const cv::Ptr<Camera> &cam2, const CameraPose &p1, const CameraPose &p2) :
            camera1(cam1), camera2(cam2), pose1(p1), pose2(p2) { }

    double triangulateIteratively(const std::vector<cv::Point2f> &points1, const std::vector<cv::Point2f> &points2,
                                  std::vector<CloudPoint> &pointcloud);

    double triangulateCv(const std::vector<cv::Point2f> &points1, const std::vector<cv::Point2f> &points2,
                         std::vector<CloudPoint> &pointcloud);

};

/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
cv::Mat_<double> LinearLSTriangulation(cv::Point3d u,        //homogenous image point (u,v,1)
                                       cv::Matx34d P,        //camera 1 matrix
                                       cv::Point3d u1,       //homogenous image point in 2nd camera
                                       cv::Matx34d P1        //camera 2 matrix
);

/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
cv::Mat_<double> IterativeLinearLSTriangulation(cv::Point3d u,    //homogenous image point (u,v,1)
                                                cv::Matx34d P,    //camera 1 matrix
                                                cv::Point3d u1,   //homogenous image point in 2nd camera
                                                cv::Matx34d P1    //camera 2 matrix
);

double TriangulatePoints(long frameId, const std::vector<cv::Point2f> &pt_set1,
                         const std::vector<cv::Point2f> &pt_set2,
                         const cv::Mat &K,
                         const cv::Mat &Kinv,
                         const cv::Mat &distcoeff,
                         const cv::Matx34d &P,
                         const cv::Matx34d &P1,
                         Cloud &pointcloud,
                         std::vector<cv::Point> &correspImg1Pt);

bool TestTriangulation(const Cloud &pcloud, const cv::Matx34d &P, std::vector<uchar> &status);
