#include <iostream>

#include "Calibration.hpp"
#include "RealStereoCamera.hpp"

using namespace std;

int main() {
    Calibration calibration("intrinsics_1013.yml", "extrinsics_1013.yml");
    RealStereoCamera rsc(&calibration);

    while (true) {
        imshow("left", rsc.getLeft());
        imshow("right", rsc.getRight());

        char ch = waitKey(50);
        if (ch != -1) {
            if (ch == 27) {
                break;
            }
        }
    }

    return 0;
}
