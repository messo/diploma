#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include "Calibration.h"

using namespace cv;
using namespace std;

bool Calibration::acquireFrames() {
    Mat frame;
    camera->read(frame);

    Mat grayscale;
    cvtColor(frame, grayscale, COLOR_BGR2GRAY);

    vector<Point2f> corners;
    bool found = findChessboardCorners(grayscale, boardSize, corners,
                                       CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
    if (!found) {
        return false;
    }

    cornerSubPix(grayscale, corners, Size(11, 11), Size(-1, -1),
                 TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 30, 0.01));

    frame.copyTo(lastFrame);
    everyCorners.push_back(corners);

    return true;
}

void Calibration::drawChessboardCorners(Mat &image) {
    lastFrame.copyTo(image);
    cv::drawChessboardCorners(image, boardSize, everyCorners.back(), true);
}

bool Calibration::calibrate() {

    vector<vector<Point3f>> objectPoints;
    objectPoints.resize(everyCorners.size());

    for (int i = 0; i < everyCorners.size(); i++) {
        for (int j = 0; j < boardSize.height; j++)
            for (int k = 0; k < boardSize.width; k++)
                objectPoints[i].push_back(Point3f(j * squareSize, k * squareSize, 0));
    }

    calibrateCamera(objectPoints, everyCorners, Size(640, 480), cameraMatrix, distCoeffs, rvecs, tvecs);

    return true;
}

void Calibration::save(const std::string &calibrationFile) {
    FileStorage fs;
    fs.open(calibrationFile, FileStorage::WRITE);

    if (camera->getId() == Camera::LEFT) {
        fs << "M1" << cameraMatrix;
        fs << "D1" << distCoeffs;
    } else {
        fs << "M2" << cameraMatrix;
        fs << "D2" << distCoeffs;
    }
}
