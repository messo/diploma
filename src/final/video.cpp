#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "camera/RealCamera.hpp"
#include "Common.h"

using namespace cv;
using namespace std;

int main() {
    RealCamera camera1(1);              // Open input
    RealCamera camera2(2);              // Open input

    Mat src1, src2;

    int i = 0;

    while (true) {
        camera1.grab();
        camera2.grab();

        camera1.retrieve(src1);
        camera2.retrieve(src2);

//        Mat img1, img2;
//        src1.copyTo(img1);
//        rectangle(img1, LEFT_SHIFT, LEFT_SHIFT + Point2f(SIZE.width, SIZE.height), Scalar(0, 0, 255), 2, LINE_AA);
//        src2.copyTo(img2);
//        rectangle(img2, RIGHT_SHIFT, RIGHT_SHIFT + Point2f(SIZE.width, SIZE.height), Scalar(0, 0, 255), 2, LINE_AA);

        imshow("1", src1);
        imshow("2", src2);

        imwrite("left_" + to_string(i) + ".png", src1);
        imwrite("right_" + to_string(i) + ".png", src2);

        i++;

        char ch = waitKey(33);
        if (ch == 27) {
            break;
        }
    }

    return 0;
}