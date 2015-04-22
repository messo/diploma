#include "FPSCounter.h"

FPSCounter::FPSCounter() {
    tickCount = cv::getTickCount();
}

void FPSCounter::tick() {
    double duration = (((double) cv::getTickCount() - tickCount) / cv::getTickFrequency());

    _avgdur = 0.98 * _avgdur + 0.02 * duration;

    if ((((double) cv::getTickCount() - _fpsstart) / cv::getTickFrequency()) > 1.0) {
        _fpsstart = cv::getTickCount();
        _avgfps = 0.7 * _avgfps + 0.3 * _fps1sec;
        _fps1sec = 0;
    }
    _fps1sec++;
}
