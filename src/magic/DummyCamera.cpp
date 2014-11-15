#include <opencv2/imgcodecs.hpp>
#include "DummyCamera.hpp"

using namespace cv;

DummyCamera::DummyCamera(int id) : Camera(id) {

}

bool DummyCamera::read(OutputArray out) {
    char filename[80];

    Mat img;
    if (getId() == 0) {
        img = imread("/home/balint/images/left_" + std::to_string(frame++));
    } else if (getId() == 1) {
        img = imread("/home/balint/images/right_" + std::to_string(frame++));
    }
    img.copyTo(out);

    if (frame > 200) {
        frame = 0;
    }

    return true;
}
