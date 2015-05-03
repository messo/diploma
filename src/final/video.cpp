#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "camera/RealCamera.hpp"

using namespace cv;
using namespace std;

int main() {
    RealCamera camera1(0);              // Open input
    RealCamera camera2(1);              // Open input

    Mat src1, src2;

    int i = 0;

    while (true) {
        camera1.grab();
        camera2.grab();

        camera1.retrieve(src1);
        camera2.retrieve(src2);

        imshow("1", src1);
        imshow("2", src2);

        imwrite("/media/balint/Data/Linux/diploma/scene_1/left_" + to_string(i) + ".png", src1);
        imwrite("/media/balint/Data/Linux/diploma/scene_1/right_" + to_string(i) + ".png", src2);

        i++;

        char ch = waitKey(33);
        if (ch == 27) {
            break;
        }
    }

    return 0;
}