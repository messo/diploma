#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include "DummyCamera.hpp"

using namespace cv;

DummyCamera::DummyCamera(int id, std::string path, int count) : Camera(id), path(path), lastFrame(count) {
}

bool DummyCamera::read(OutputArray out) {

    currentFrame = frame;

    Mat img;
    if (getId() == Camera::LEFT) {
        img = imread(path + "/left_" + std::to_string(frame) + ".png");
    } else if (getId() == Camera::RIGHT) {
        img = imread(path + "/right_" + std::to_string(frame) + ".png");
    }
    img.copyTo(out);

    frame += 2; // 3;

    if (frame > lastFrame) {
        frame = firstFrame;
    }

    return true;
}

bool DummyCamera::grab() {
    return true;
}

bool DummyCamera::retrieve(OutputArray img) {
    return read(img);
}

long DummyCamera::getFrameId() const {
    return currentFrame;
}
