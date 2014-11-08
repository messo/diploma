#include "StereoCamera.hpp"

StereoCamera::StereoCamera(Calibration *calibration) :
        calibration(calibration), imageSize(640, 480) {
    if (calibration != NULL) {
        int numberOfDisparities = ((imageSize.width / 8) + 15) & -16;
        sgbm = createStereoSGBM(0, numberOfDisparities, 3, 8 * 3 * 3 * 3, 32 * 3 * 3 * 3, 1, 63, 10, 500, 32, StereoSGBM::MODE_HH);
        dispRoi = getValidDisparityROI(calibration->validRoiLeft, calibration->validRoiRight, 0, numberOfDisparities, 3);
    }
};

Mat StereoCamera::rectify(const Mat &img, int cam) {
    Mat remapped;
    remap(img, remapped, calibration->rmap[cam][0], calibration->rmap[cam][1], INTER_LINEAR);
    return remapped;
}

Mat StereoCamera::getDisparityMatrix(Mat &left, Mat &right) {
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
