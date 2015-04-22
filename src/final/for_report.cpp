#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {

    Mat img = imread("/media/balint/Data/cucc/mask343.png");
    Mat mask;
    cvtColor(img, mask, COLOR_RGB2GRAY);

    Mat maskWithoutShadows = mask.clone();
    for (int y = 0; y < maskWithoutShadows.rows; y++) {
        for (int x = 0; x < maskWithoutShadows.cols; x++) {
            if (maskWithoutShadows.at<uchar>(y, x) != 255) {
                maskWithoutShadows.at<uchar>(y, x) = 0;
            }
        }
    }

    Mat result = maskWithoutShadows.clone();

    int niters = 3;

    Mat small = getStructuringElement(MORPH_RECT, Size(2, 2));
    Mat bigger = getStructuringElement(MORPH_RECT, Size(5, 5));

    erode(result, result, small, Point(-1, -1), niters);
    dilate(result, result, bigger, Point(-1, -1), niters * 2);
    erode(result, result, bigger, Point(-1, -1), niters * 2);

//    erode(result, result, Mat(), Point(-1, -1), niters * 2);
//    dilate(result, result, Mat(), Point(-1, -1), niters);
//    erode(result, result, Mat(), Point(-1, -1), niters);

    imwrite("/media/balint/Data/cucc/mask343_fixed.png", result);

    Mat original = imread("/media/balint/Data/cucc/image343.png");
    Mat applied;
    original.copyTo(applied, result);
    imwrite("/media/balint/Data/cucc/mask343_applied.png", applied);






    vector<vector<Point>> contours;

    findContours(result, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat contourResult(480, 640, CV_8U, Scalar(0));

    if (contours.size() != 0) {
        // iterate through all the top-level contours,
        // draw each connected component with its own random color
        int largestComp = 0;
        double maxArea = 0;

        for (int idx = 0; idx < contours.size(); idx++) {
            const vector<Point> &c = contours[idx];
            double area = fabs(contourArea(Mat(c)));
            if (area > maxArea) {
                maxArea = area;
                largestComp = idx;
            }
        }

        for (int idx = 0; idx < contours.size(); idx++) {
            if (idx == largestComp) {
                drawContours(contourResult, contours, idx, Scalar(255), FILLED, LINE_8);
            }  else {
                drawContours(contourResult, contours, idx, Scalar(255), 1, LINE_AA);
            }
        }
    }

    imwrite("/media/balint/Data/cucc/mask343_contours.png", contourResult);

    while (true) {
        imshow("original", mask);
        imshow("maskWithoutShadows", maskWithoutShadows);
        imshow("result", result);
        imshow("contourResult", contourResult);

        char ch = waitKey(50);
        if (ch == 27) {
            break;
        }
    }

    return 0;
}