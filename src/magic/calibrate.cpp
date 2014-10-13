#include "Calibration.hpp"
#include "RealStereoCamera.hpp"

/* Keep the webcam from locking up when you interrupt a frame capture */
volatile int quit_signal = 0;
#ifdef __unix__

#include <signal.h>

extern "C" void quit_signal_handler(int signum) {
    if (quit_signal != 0) exit(0); // just exit already
    quit_signal = 1;
    printf("Will quit at next camera frame (repeat to kill now)\n");
}
#endif

int main() {
    RealStereoCamera rsc;
    Calibration calibration(NULL, NULL);

#ifdef __unix__
    signal(SIGINT, quit_signal_handler); // listen for ctrl-C
#endif

    while (true) {
        imshow("left", rsc.getLeft());
        imshow("right", rsc.getRight());

        char ch = waitKey(50);
        if (ch != -1) {
            if (ch == ' ') {
                calibration.acquireFrames(rsc);
                continue;
            } else if (ch == 'c') {
                calibration.calibrate();
                break;
            } else if (ch == 27) {
                break;
            }
        }

        if (quit_signal) exit(0);
    }
}