#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include "CameraPoseCalculator.h"
#include "../camera/Camera.hpp"

using namespace cv;

bool CameraPoseCalculator::calculate() {
    Size boardSize(9, 6);
    float squareSize = 1.0f;

    Mat img;
    if (!camera->K.empty()) {
        camera->readUndistorted(img);
    } else {
        camera->read(img);
    }

    Mat grayscale;
    cvtColor(img, grayscale, COLOR_BGR2GRAY);

    std::vector<Point2f> imagePoints;

    int maxScale = 3;
    bool found = false;
    for (int scale = 1; scale <= maxScale; scale++) {
        Mat scaledImg;
        if (scale == 1)
            scaledImg = grayscale;
        else
            resize(grayscale, scaledImg, Size(), scale, scale);
        found = findChessboardCorners(scaledImg, boardSize, imagePoints,
                                      CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);
        if (found) {
            cornerSubPix(scaledImg, imagePoints, Size(11, 11), Size(-1, -1),
                         TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 30, 0.01));

            if (scale > 1) {
                Mat cornersMat(imagePoints);
                cornersMat *= 1. / scale;
            }
            break;
        }
    }

    if (!found) {
        std::cout << "Chessboard not found. Pose cannot be estimated!";
        cameraPose = Ptr<CameraPose>();
        return false;
    }

    std::vector<Point3f> objectPoints;
    for (int j = 0; j < boardSize.height; j++)
        for (int k = 0; k < boardSize.width; k++)
            objectPoints.push_back(Point3f(j * squareSize, k * squareSize, 0));

    cameraPose = Ptr<CameraPose>(new CameraPose());
    solvePnP(objectPoints, imagePoints, camera->K, camera->distCoeffs, cameraPose->rvec, cameraPose->tvec);

    return true;
}

bool CameraPoseCalculator::poseCalculated() {
    return !cameraPose.empty();
}
