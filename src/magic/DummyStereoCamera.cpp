#include "DummyStereoCamera.hpp"

Mat DummyStereoCamera::getLeft() {
    frameId++;
    if (frameId > 200) {
        frameId = 0;
    }
    return getImage(0);
}

Mat DummyStereoCamera::getRight() {
    return getImage(1);
}

Mat DummyStereoCamera::getImage(int cam) {
    char filename[80];
    if (cam == 0) {
        sprintf(filename, "/home/balint/images/left_%d.png", frameId);
    } else if (cam == 1) {
        sprintf(filename, "/home/balint/images/right_%d.png", frameId);
    }
    Mat img = imread(filename);

    if (calibration != NULL) {
        return rectify(img, cam);
    } else {
        return img;
    }
}
