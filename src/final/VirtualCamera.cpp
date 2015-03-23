#include "VirtualCamera.h"

using namespace cv;

VirtualCamera::VirtualCamera() {
    tvec = (Mat_<double>(1, 3) << 0, 0, 0);
    rvec = (Mat_<double>(1, 3) << 0, 0, 0);
}

void VirtualCamera::setRotation(int rot1, int rot2, int rot3) {
    double x = (double) rot1 / 180.0 * M_PI;
    double y = (double) rot2 / 180.0 * M_PI;
    double z = (double) rot3 / 180.0 * M_PI;

    Mat rx = (Mat_<double>(3, 3) << 1, 0, 0, 0, cos(x), -sin(x), 0, sin(x), cos(x));

    Mat ry = (Mat_<double>(3, 3) << cos(y), 0, sin(y), 0, 1, 0, -sin(y), 0, cos(y));

    Mat rz =
            (Mat_<double>(3, 3) << cos(z), -sin(z), 0, sin(z), cos(z), 0, 0, 0, 1);

    Mat ret;
    Rodrigues(Mat::eye(3, 3, CV_64F) * rx * ry * rz, rvec);
}

void VirtualCamera::rotX(int deg) {
    Mat rmat;
    Rodrigues(rvec, rmat);

    double x = (double) deg / 180.0 * M_PI;
    Mat rx = (Mat_<double>(3, 3) << 1, 0, 0, 0, cos(x), -sin(x), 0, sin(x), cos(x));

    Rodrigues(rmat * rx, rvec);
}

void VirtualCamera::rotY(int deg) {
    Mat rmat;
    Rodrigues(rvec, rmat);

    double y = (double) deg / 180.0 * M_PI;
    Mat ry = (Mat_<double>(3, 3) << cos(y), 0, sin(y), 0, 1, 0, -sin(y), 0, cos(y));

    Rodrigues(rmat * ry, rvec);
}

void VirtualCamera::rotZ(int deg) {
    Mat rmat;
    Rodrigues(rvec, rmat);

    double z = (double) deg / 180.0 * M_PI;
    Mat rz =
            (Mat_<double>(3, 3) << cos(z), -sin(z), 0, sin(z), cos(z), 0, 0, 0, 1);

    Rodrigues(rmat * rz, rvec);
}

void VirtualCamera::setTranslation(double x, double y, double z) {
    tvec = (Mat_<double>(1, 3) << x, y, z);
}

void VirtualCamera::addX(double x) {
    tvec.at<double>(0, 0) += x;
}

void VirtualCamera::addY(double y) {
    tvec.at<double>(0, 1) += y;
}

void VirtualCamera::addZ(double z) {
    tvec.at<double>(0, 2) += z;
}

VirtualCamera::~VirtualCamera() {
    // TODO Auto-generated destructor stub
}

