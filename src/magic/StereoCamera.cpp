#include "StereoCamera.hpp"

Mat StereoCamera::rectify(const Mat &img, int cam) {
    Mat remapped;
    remap(img, remapped, calibration->rmap[cam][0], calibration->rmap[cam][1], INTER_LINEAR);
    return remapped;
}
