#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <iomanip>
#include <sstream>
#include "DinoCamera.hpp"

using namespace cv;

DinoCamera::DinoCamera()
        : Camera(Camera::LEFT), path("/media/balint/Data/Linux/dinoRing"), lastFrame(47) {
}

bool DinoCamera::read(OutputArray out) {

    std::ostringstream ss;
    ss << std::setw(4) << std::setfill('0') << (frame++);

    Mat img;
    img = imread(path + "/dinoR" + ss.str() + ".png");
    std::cout << path + "/dinoR" + ss.str() + ".png" << std::endl;
    std::cout.flush();
    img.copyTo(out);

    if (frame > lastFrame) {
        frame = firstFrame;
    }

    return true;
}

bool DinoCamera::grab() {
    return true;
}

bool DinoCamera::retrieve(OutputArray img) {
    return read(img);
}
