/*
 * Magic.h
 *
 *  Created on: 2014.02.28.
 *      Author: Balint
 */

#ifndef MAGIC_H_
#define MAGIC_H_

#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Camera.h"
#include "PointCloudSegmenter.h"

using namespace cv;
using namespace std;

const int LEFT = 0;
const int RIGHT = 1;
const float MAX_Z = 20.0f;

enum SC_TYPE {
    BM, SGBM, VAR
};

class Magic {
    Size imageSize;
    StereoBM *bm;
    StereoSGBM *sgbm;
    Mat cameraMatrix[2], distCoeffs[2];
    Mat R, T, Q;
    Mat rmap[2][2];
public:
    Mat P1, P2, E, F;
    vector<Point3f> objectPoints;
    vector<Point2f> imagePoints;
    Rect validRoiLeft, validRoiRight, dispRoi;

    Magic();

    virtual ~Magic();

    Mat readAndRemap(const string &filename, int cam);

    Mat readAndRemap(VideoCapture &cap, int cam);

    void remap(const Mat &input, Mat &output, int cam);

    void drawEpipolarLines(Mat &canvas);

    void getDisparityMatrix(const Mat &left, const Mat &right, SC_TYPE type, Mat &magic);

    Mat normalizeDisparity(const Mat &imgDisparity16S);

    void reprojectTo3D(const Mat &imgDisparity16S, Mat &xyz);

    void filter3DCoordinates(Mat &xyz, const Rect &roi);

    void getCameraPose(Mat &rvec, Mat &tvec);

    Mat getGlProjectionMatrix();

    void getPointsForTransformation(vector<Point3f> &objs, vector<Point2f> &pixels, const Camera &camera);

    void projectPoints(const vector<Point3f> &objectPoints, const Camera &camera, vector<Point2f> &imagePoints);

    void projectPoints(const Camera &camera, vector<Point2f> &imagePoints);

    void reprojectPoints(const Mat &rvec, const Mat &tvec, const Mat &img, Mat &output);

    void drawTest(Mat &img, const Mat &left, const Camera &camera);
};

#endif /* MAGIC_H_ */
