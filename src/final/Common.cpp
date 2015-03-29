#include "Common.h"

#include <fstream>
#include <opencv2/core/mat.hpp>

using namespace cv;
using namespace std;

vector<Point3d> CloudPointsToPoints(const vector<CloudPoint> &cpts) {
    vector<Point3d> out;
    for (unsigned int i = 0; i < cpts.size(); i++) {
        out.push_back(cpts[i].pt);
    }
    return out;
}

void writeCloudPoints(const vector<CloudPoint> &cpts) {
    ofstream myfile;

    myfile.open("/media/balint/Data/Linux/cloud.ply");
    myfile << "ply " << endl;
    myfile << "format ascii 1.0" << endl;
    myfile << "element vertex " << cpts.size() << endl;
    myfile << "property float x" << endl;
    myfile << "property float y" << endl;
    myfile << "property float z" << endl;
    myfile << "end_header" << endl;
    for (unsigned int i = 0; i < cpts.size(); i++) {
        myfile << cpts[i].pt.x << " " << cpts[i].pt.y << " " << cpts[i].pt.z << endl;
    }
    myfile.close();
}

void translate(const std::vector<cv::Point> &input, cv::Point translation, std::vector<cv::Point> &output) {
    for (int i = 0; i < input.size(); i++) {
        output.push_back(input[i] + translation);
    }
}

void translate(std::vector<cv::Point> &input, cv::Point translation) {
    for (int i = 0; i < input.size(); i++) {
        input[i] += translation;
    }
}

Mat mergeImages(const Mat &left, const Mat &right) {
    Mat result = Mat::zeros(left.rows, left.cols + right.cols, left.type());

    left.copyTo(result(Rect(0, 0, left.cols, left.rows)));
    right.copyTo(result(Rect(left.cols - 1, 0, right.cols, right.rows)));

    return result;
}

void shiftImage(const cv::Mat &input, const cv::Rect &boundingRect,
                const cv::Point2i &translation, cv::Mat &output) {
    output.setTo(0);
    input(boundingRect).copyTo(output(boundingRect + translation));
}
