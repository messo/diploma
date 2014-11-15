#include "StereoCamera.hpp"
#include "RealCamera.hpp"

#define FPS_ENABLED

using namespace cv;

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
        // NOT IMPLEMENTED YET!
    }

    if (calibration.get() != NULL) {
        int numberOfDisparities = ((imageSize.width / 8) + 15) & -16;
        P1 = 8 * 3 * 3 * 3;
        P2 = 32 * 3 * 3 * 3;
        preFilterCap = 63;
        uniquenessRatio = 10;
        speckleWindowSize = 500;
        speckleRange = 32;
        SADWindowSize = 10;
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

Mat StereoCamera::normalizeDisparity(const Mat &imgDisparity16S) {
    Mat imgDisparity8U = Mat(imgDisparity16S.rows, imgDisparity16S.cols, CV_8UC1);
    double minVal;
    double maxVal;
    minMaxLoc(imgDisparity16S, &minVal, &maxVal);
    imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 255 / (maxVal - minVal));

    return imgDisparity8U;
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

cv::Mat StereoCamera::getLeft() {
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

cv::Mat StereoCamera::getRight() {
    return getImage(rightCamera);
}
