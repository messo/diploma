#include "RealStereoCamera.hpp"

Mat RealStereoCamera::getLeft() {
    return getImage(leftCamera, 0);
}

Mat RealStereoCamera::getRight() {
    return getImage(rightCamera, 1);
}

Mat RealStereoCamera::getImage(VideoCapture &capture, int cam) {
    Mat img;
    capture.read(img);

    if (calibration != NULL) {
        return rectify(img, cam);
    } else {
        return img;
    }
}
