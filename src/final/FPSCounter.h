#pragma once

// logic from: http://stackoverflow.com/questions/22148826/measure-opencv-fps
// thanks to Zaw Lin

#include <opencv2/core/utility.hpp>

class FPSCounter {

private:
    double tickCount;

    double _avgdur = 0;
    double _fpsstart = 0;
    double _avgfps = 0;
    double _fps1sec = 0;

public:

    FPSCounter();

    void tick();

    double get() const {
        return _avgfps;
    }
};
