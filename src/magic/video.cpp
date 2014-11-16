#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "RealCamera.hpp"

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

        char filename1[80];
        sprintf(filename1,"/media/balint/Data/Linux/diploma/src/images1/left_%d.png",i);
        imwrite(filename1, src1);

        char filename2[80];
        sprintf(filename1,"/media/balint/Data/Linux/diploma/src/images1/right_%d.png",i);
        imwrite(filename1, src2);

        i++;

        char ch = waitKey(50);
        if (ch == 27) {
            break;
        }
    }

    return 0;
}