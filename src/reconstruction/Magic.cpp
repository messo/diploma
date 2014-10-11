/*
 * Magic.cpp
 *
 *  Created on: 2014.02.28.
 *      Author: Balint
 */

#include "Magic.h"

Magic::Magic() :
//      imageSize(640, 480), sbm(StereoBM::BASIC_PRESET,
//              16 * 5 /**< Range of disparity */,
//              5 /**< Size of the block window. Must be odd */) {
        imageSize(640, 480) {

    int numberOfDisparities = ((imageSize.width / 8) + 15) & -16;
    //int SADWindowSize = 0;

    //namedWindow("Magic", 1);
    //createTrackbar("preFilterCap", "Magic", &(bm.state->preFilterCap), 63);
    //createTrackbar("SADWindowSize", "Magic", &(bm.state->SADWindowSize), 21);
    //createTrackbar("textureThreshold", "Magic", &(bm.state->textureThreshold),
//          100);
//  createTrackbar("uniquenessRatio", "Magic", &(bm.state->uniquenessRatio),
//          50);
//  createTrackbar("speckleWindowSize", "Magic", &(bm.state->speckleWindowSize),
//          200);
//  createTrackbar("speckleRange", "Magic", &(bm.state->speckleRange), 128);

    bm = createStereoBM(numberOfDisparities, 13);

    /*bm.state->roi1 = ((CvRect)validRoiLeft);
    bm.state->roi2 = ((CvRect)validRoiRight);
    bm.state->preFilterCap = 31;
    bm.state->SADWindowSize = 13;
    bm.state->minDisparity = 0;
    bm.state->numberOfDisparities = numberOfDisparities;
    bm.state->textureThreshold = 10;
    bm.state->uniquenessRatio = 1;
    bm.state->speckleWindowSize = 100;
    bm.state->speckleRange = 32;
    bm.state->disp12MaxDiff = 1;*/

//  namedWindow("Magic SGBM", 1);
//  createTrackbar("preFilterCap", "Magic SGBM", &(sgbm.preFilterCap), 100);
//  createTrackbar("SADWindowSize", "Magic SGBM", &(sgbm.SADWindowSize), 21);
//  createTrackbar("uniquenessRatio", "Magic SGBM", &(sgbm.uniquenessRatio),
//          50);
//  createTrackbar("speckleWindowSize", "Magic SGBM", &(sgbm.speckleWindowSize),
//          2000);
//  createTrackbar("speckleRange", "Magic SGBM", &(sgbm.speckleRange), 2000);
//  createTrackbar("P1", "Magic SGBM", &(sgbm.P1), 2000);
//  createTrackbar("P2", "Magic SGBM", &(sgbm.P2), 2000);

    sgbm = createStereoSGBM(0, numberOfDisparities, 3, 8 * 3 * 3 * 3, 32 * 3 * 3 * 3, 1, 63, 10, 500, 32, StereoSGBM::MODE_HH);

    //namedWindow("VAR", 1);
    //createTrackbar("levels", "VAR", &(var.levels), 10);
    //createTrackbar("nIt", "VAR", &(var.nIt), 100);
    //createTrackbar("poly_n", "VAR", &(var.poly_n), 10);

    /*var.levels = 3;                              // ignored with USE_AUTO_PARAMS
    var.pyrScale = 0.5;                          // ignored with USE_AUTO_PARAMS
    var.nIt = 25;
    var.minDisp = -numberOfDisparities;
    var.maxDisp = 0;
    var.poly_n = 3;
    var.poly_sigma = 0.0;
    //var.fi = 15.0f;
    //var.lambda = 0.03f;
    var.penalization = var.PENALIZATION_TICHONOV; // ignored with USE_AUTO_PARAMS
    var.cycle = var.CYCLE_V;                     // ignored with USE_AUTO_PARAMS
    var.flags = var.USE_SMART_ID | var.USE_AUTO_PARAMS | var.USE_INITIAL_DISPARITY | var.USE_MEDIAN_FILTERING;*/

    cameraMatrix[0] = Mat::eye(3, 3, CV_64F);
    cameraMatrix[1] = Mat::eye(3, 3, CV_64F);

    // save intrinsic parameters
    FileStorage fs("intrinsics.yml", CV_STORAGE_READ);
    if (fs.isOpened()) {
        fs["M1"] >> cameraMatrix[0];
        fs["D1"] >> distCoeffs[0];
        fs["M2"] >> cameraMatrix[1];
        fs["D2"] >> distCoeffs[1];
        fs.release();
    } else
        cout << "Error: can not load the intrinsic parameters\n";

    Mat R1, R2;
    // Rect validRoi[2];

    fs.open("extrinsics.yml", CV_STORAGE_READ);
    if (fs.isOpened()) {
        fs["R"] >> R;
        fs["T"] >> T;
        fs["E"] >> E;
        fs["F"] >> F;
        fs["R1"] >> R1;
        fs["R2"] >> R2;
        fs["P1"] >> P1;
        fs["P2"] >> P2;
        fs["Q"] >> Q;
        fs["validRoiLeft"] >> validRoiLeft;
        fs["validRoiRight"] >> validRoiRight;
        fs.release();
    } else
        cout << "Error: can not load the intrinsic parameters\n";

    //Precompute maps for cv::remap()
    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize,
            CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize,
            CV_16SC2, rmap[1][0], rmap[1][1]);

    dispRoi = getValidDisparityROI(validRoiLeft, validRoiRight, sgbm->getMinDisparity(), sgbm->getNumDisparities(), sgbm->getBlockSize());
}

Mat Magic::readAndRemap(const string &filename, int cam) {
    Mat remapped;
    remap(imread(filename), remapped, cam);
    return remapped;

//  std::cout << remapped.cols << ", " << remapped.rows << "\n";
//  std::cout << validRoiLeft.x << ", " << validRoiLeft.y << "; "
//          << validRoiLeft.width << ", " << validRoiLeft.height << "\n\n";
//  std::cout.flush();
//  Mat result = remapped((cam == LEFT) ? validRoiLeft : validRoiRight);
//  return result;
}

Mat Magic::readAndRemap(VideoCapture &cap, int cam) {
    Mat img, remapped;
    cap.read(img);
    remap(img, remapped, cam);
    return remapped;
}

void Magic::remap(const Mat &input, Mat &output, int cam) {
    cv::remap(input, output, rmap[cam][0], rmap[cam][1], INTER_LINEAR);
}

void Magic::drawEpipolarLines(Mat &canvas) {
    for (int j = 0; j < canvas.rows; j += 16)
        line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
}

void Magic::getDisparityMatrix(const Mat &left, const Mat &right, SC_TYPE type, Mat &magic) {
    if (type == VAR) {
        magic.create(left.rows, left.cols, CV_8UC1);
    } else {
        magic.create(left.rows, left.cols, CV_16S);
    }

    // ---------
    // DISPARITY
    // ---------
    Mat imgLeftGray, imgRightGray;
    cvtColor(left, imgLeftGray, COLOR_BGR2GRAY);
    cvtColor(right, imgRightGray, COLOR_BGR2GRAY);

    sgbm->compute(imgLeftGray, imgRightGray, magic);
    //bm(imgLeftGray, imgRightGray, magic);
    //var(imgLeftGray, imgRightGray, magic);
}

void Magic::reprojectTo3D(const Mat &imgDisparity16S, Mat &xyz) {
    reprojectImageTo3D(imgDisparity16S, xyz, Q, true);
}

Mat Magic::normalizeDisparity(const Mat &imgDisparity16S) {
    Mat imgDisparity8U = Mat(imgDisparity16S.rows, imgDisparity16S.cols, CV_8UC1);
    double minVal;
    double maxVal;
    minMaxLoc(imgDisparity16S, &minVal, &maxVal);
    imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 255 / (maxVal - minVal));

    return imgDisparity8U;
}

void Magic::filter3DCoordinates(Mat &xyz, const Rect &roi) {
    objectPoints.clear();
    imagePoints.clear();
    for (int y = 0; y < xyz.rows; y++) {
        for (int x = 0; x < xyz.cols; x++) {
            Vec3f &point = xyz.at<Vec3f>(y, x);
            if (!roi.contains(Point(x, y)) || fabs(point[2] - MAX_Z) < FLT_EPSILON || fabs(point[2]) > MAX_Z) {
                point[2] = -1.0f;
            } else {
                objectPoints.push_back(Point3f(point));
                imagePoints.push_back(Point2f((float) x, (float) y));
            }
        }
    }
}

void Magic::getCameraPose(Mat &rvec, Mat &tvec) {
    solvePnP(objectPoints, imagePoints, cameraMatrix[LEFT], distCoeffs[LEFT], rvec, tvec);
}

Mat Magic::getGlProjectionMatrix() {
    Mat &cam = cameraMatrix[LEFT];

    Mat projMat = Mat::zeros(4, 4, CV_64FC1);
    float far = 100.0f, near = 0.5f;
    projMat.at<double>(0, 0) = 2 * cam.at<double>(0, 0) / 640.;
    projMat.at<double>(0, 2) = -1 + (2 * cam.at<double>(0, 2) / 640.);
    projMat.at<double>(1, 1) = 2 * cam.at<double>(1, 1) / 480.;
    projMat.at<double>(1, 2) = -1 + (2 * cam.at<double>(1, 2) / 480.);
    projMat.at<double>(2, 2) = -(far + near) / (far - near);
    projMat.at<double>(2, 3) = -2 * far * near / (far - near);
    projMat.at<double>(3, 2) = -1;

    return projMat;
}

void Magic::getPointsForTransformation(vector<Point3f> &objs, vector<Point2f> &pixels, const Camera &camera) {
    unsigned int size = objectPoints.size();
    vector<Point2f> _imagePoints;
    projectPoints(camera, _imagePoints);

    std::cout << "All: " << size << " " << _imagePoints.size() << "\n";

    for (unsigned int i = 0; i < _imagePoints.size(); i++) {
        int y = (int) _imagePoints[i].y;
        int x = (int) _imagePoints[i].x;
        if (x >= 0 && y >= 0 && y < imageSize.width && x < imageSize.height) {
            objs.push_back(objectPoints[i]);
            pixels.push_back(imagePoints[i]);
        }
    }

    std::cout << "Értékes: " << objs.size() << "\n";

    std::cout.flush();
}

void Magic::projectPoints(const vector<Point3f> &objectPoints, const Camera &camera, vector<Point2f> &imagePoints) {
    cv::projectPoints(objectPoints, camera.getRVec(), camera.getTVec(), cameraMatrix[LEFT], distCoeffs[LEFT],
            imagePoints);
}

void Magic::projectPoints(const Camera &camera, vector<Point2f> &imagePoints) {
    cv::projectPoints(objectPoints, camera.getRVec(), camera.getTVec(), cameraMatrix[LEFT], distCoeffs[LEFT],
            imagePoints);
}

void Magic::reprojectPoints(const Mat &rvec, const Mat &tvec, const Mat &img, Mat &output) {
    Mat points;
    cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix[LEFT], distCoeffs[LEFT],
            points);

    for (int i = 0; i < points.rows; i++) {
        Mat point = points.row(i);
        int x = (int) point.at<float>(0, 0);
        int y = (int) point.at<float>(0, 1);
        if (y >= 0 && x >= 0 && y < img.rows && x < img.cols) {
            const Vec3b &orig = img.at<Vec3b>(y, x);
            Vec3b &vec = output.at<Vec3b>(y, x);
            vec[0] = orig[0];
            vec[1] = orig[1];
            vec[2] = orig[2];
        }
    }
}

void Magic::drawTest(Mat &img, const Mat &left, const Camera &camera) {
    vector<Point2f> _imagePoints;
    projectPoints(camera, _imagePoints);
    for (unsigned int i = 0; i < _imagePoints.size(); i++) {
        int y = (int) _imagePoints[i].y;
        int x = (int) _imagePoints[i].x;
        if (x >= 0 && y >= 0 && y < img.rows && x < img.cols)
            img.at<Vec3b>(y, x) = left.at<Vec3b>(imagePoints[i]);
    }
}

Magic::~Magic() {
    // TODO Auto-generated destructor stub
}
