#include "StereoCamera.hpp"
#include "RealCamera.hpp"
#include "DummyCamera.hpp"

#define FPS_ENABLED

using namespace cv;

const float MAX_Z = 5.0f;

cv::Ptr<cv::StereoSGBM> sgbm;

int preFilterCap;
int SADWindowSize;
int uniquenessRatio;
int speckleWindowSize;
int speckleRange;
int P1;
int P2;

void onChanged(int pos, void *userdata) {
    int numberOfDisparities = ((640 / 8) + 15) & -16;
    sgbm = createStereoSGBM(0, numberOfDisparities, SADWindowSize, P1, P2, 1, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, StereoSGBM::MODE_SGBM);
}

StereoCamera::StereoCamera(Type type, cv::Ptr<Calibration> calibration) :
        calibration(calibration),
        imageSize(640, 480) {
    if (type == REAL) {
        leftCamera = Ptr<Camera>(new RealCamera(0));
        rightCamera = Ptr<Camera>(new RealCamera(1));
    } else if (type == DUMMY) {
        leftCamera = Ptr<Camera>(new DummyCamera(0));
        rightCamera = Ptr<Camera>(new DummyCamera(1));
    }

    if (calibration.get() != NULL) {
        int numberOfDisparities = ((imageSize.width / 8) + 15) & -16;
        P1 = 8 * 3 * 3 * 3;
        P2 = 32 * 3 * 3 * 3;
        preFilterCap = 63;
        uniquenessRatio = 2;
        speckleWindowSize = 500;
        speckleRange = 32;
        SADWindowSize = 5;
        sgbm = createStereoSGBM(0, numberOfDisparities, SADWindowSize, P1, P2, 1, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, StereoSGBM::MODE_SGBM);

        namedWindow("Magic SGBM", 1);
        createTrackbar("preFilterCap", "Magic SGBM", &preFilterCap, 100, onChanged);
        createTrackbar("SADWindowSize", "Magic SGBM", &SADWindowSize, 21, onChanged);
        createTrackbar("uniquenessRatio", "Magic SGBM", &uniquenessRatio, 50, onChanged);
        createTrackbar("speckleWindowSize", "Magic SGBM", &speckleWindowSize, 2000, onChanged);
        createTrackbar("speckleRange", "Magic SGBM", &speckleRange, 2000, onChanged);
        createTrackbar("P1", "Magic SGBM", &P1, 2000, onChanged);
        createTrackbar("P2", "Magic SGBM", &P2, 2000, onChanged);

        dispRoi = getValidDisparityROI(calibration->validRoiLeft, calibration->validRoiRight, 0, numberOfDisparities, 3);
    }
};


Mat StereoCamera::getLeft() {
#ifdef FPS_ENABLED
    if (counter == -1) {
        time(&start);
        counter++;
    } else {
        time(&end);
        fps = (++counter) / difftime(end, start);
        printf("FPS = %.2f\n", fps);
        std::cout.flush();
    }
#endif

    return getImage(leftCamera);
}

Mat StereoCamera::getRight() {
    return getImage(rightCamera);
}

Mat StereoCamera::getImage(cv::Ptr<Camera> camera) {
    Mat img;
    camera->read(img);

    if (calibration != NULL) {
        return rectify(img, camera->getId());
    } else {
        return img;
    }
}

Mat StereoCamera::rectify(const Mat &img, int cam) {
    Mat remapped;
    remap(img, remapped, calibration->rmap[cam][0], calibration->rmap[cam][1], INTER_LINEAR);
    return remapped;
}

Mat StereoCamera::getDisparityMatrix(const Mat &left, const Mat &right) {
    Mat magic(left.rows, left.cols, CV_8UC1);

    Mat imgLeftGray, imgRightGray;
    cvtColor(left, imgLeftGray, COLOR_BGR2GRAY);
    cvtColor(right, imgRightGray, COLOR_BGR2GRAY);

    sgbm->compute(imgLeftGray, imgRightGray, magic);

    return magic;
}

Mat StereoCamera::normalizeDisparity(const Mat &disparity16S) {
    Mat disparity8U = Mat(disparity16S.rows, disparity16S.cols, CV_8UC1);
    double minVal;
    double maxVal;
    minMaxLoc(disparity16S, &minVal, &maxVal);
    disparity16S.convertTo(disparity8U, CV_8UC1, 255 / (maxVal - minVal));

    return disparity8U;
}

bool StereoCamera::reprojectTo3D(const cv::Mat &disparity16S) {
    Mat xyz;
    reprojectImageTo3D(disparity16S, xyz, calibration->Q, true);

    objectPoints.clear();
    imagePoints.clear();
    for (int y = 0; y < xyz.rows; y++) {
        for (int x = 0; x < xyz.cols; x++) {
            Vec3f &point = xyz.at<Vec3f>(y, x);
            if (!dispRoi.contains(Point(x, y)) || fabs(point[2] - MAX_Z) < FLT_EPSILON || fabs(point[2]) > MAX_Z) {
                point[2] = -1.0f;
            } else {
                objectPoints.push_back(Point3f(point));
                imagePoints.push_back(Point2f((float) x, (float) y));
            }
        }
    }

    return objectPoints.size() > 0;
}

void StereoCamera::getCameraPose(cv::OutputArray rvec, cv::OutputArray tvec) {
    solvePnP(objectPoints, imagePoints, calibration->cameraMatrix[0], calibration->distCoeffs[0], rvec, tvec);
}

void StereoCamera::reprojectPoints(const Mat &rvec, const Mat &tvec, const Mat &img, Mat &output) {
    Mat points;
    cv::projectPoints(objectPoints, rvec, tvec, calibration->cameraMatrix[0], calibration->distCoeffs[0], points);

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
