#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

int main() {
    VideoCapture inputVideo1(0);              // Open input
    VideoCapture inputVideo2(1);              // Open input

    if(!inputVideo1.isOpened()) {
        return -1;
    }

    if(!inputVideo2.isOpened()) {
        return -1;
    }

    Mat src1, src2;

    int i = 0;

    while (true) {
        inputVideo1 >> src1;              // read
        inputVideo2 >> src2;              // read


        imshow("1", src1);
        imshow("2", src2);

        char filename1[80];
        sprintf(filename1,"left_%d.png",i);
        imwrite(filename1, src1);

        char filename2[80];
        sprintf(filename1,"right_%d.png",i);
        imwrite(filename1, src1);

        i++;

        char ch = waitKey(50);
        if (ch == 27) {
            break;
        }
    }

    return 0;
}