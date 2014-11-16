#include <opencv2/imgcodecs.hpp>
#include "DummyCamera.hpp"

using namespace cv;

DummyCamera::DummyCamera(int id) : Camera(id) {

}

bool DummyCamera::read(OutputArray out) {
    char filename[80];

    Mat img;
    if (getId() == 0) {
        img = imread("/media/balint/Data/Linux/diploma/src/images1/left_" + std::to_string(frame++) + ".png");
    } else if (getId() == 1) {
        img = imread("/media/balint/Data/Linux/diploma/src/images1/right_" + std::to_string(frame++) + ".png");
    }
    img.copyTo(out);

    if (frame > 231) {
        frame = 0;
    }

    return true;
}

bool DummyCamera::grab() {
    return true;
}

bool DummyCamera::retrieve(OutputArray img) {
    return read(img);
}
