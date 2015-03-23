//
// Created by balint on 2015.03.22..
//

#include <iostream>
#include <opencv2/core/types.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "TriTest.h"
#include "VirtualCamera.h"
#include "Triangulation.h"
#include "OFReconstruction.h"
#include "FakeCamera.h"

using namespace std;
using namespace cv;

int main() {

    // ---- DATAS

    vector<Point3f> pt_3d;

    for(float x=-10; x<=10; x += 0.1) {
        for (float y = 0; y > -20; y -= 0.5) {
            pt_3d.push_back(Point3f(x, y, 5));
        }
    }

    for (float x = -10; x <= 10; x += 0.1) {
        for (float z = 0; z <= 5; z += 0.1) {
            pt_3d.push_back(Point3f(x, 0, z));
        }
    }

    for (float y = 0; y > -20; y -= 0.5) {
        for (float z = 0; z <= 5; z += 0.1) {
            pt_3d.push_back(Point3f(-10, y, z));
        }
    }

    // ---- CAMERA

    VirtualCamera vc;
    Ptr<FakeCamera> fc(new FakeCamera(Camera::LEFT));

    vc.addY(9);
    vc.addZ(1);

    vector<Point2f> reprojected_pt_set1, reprojected_pt_set2;

    vc.addX(-4);
    projectPoints(pt_3d, vc.getRVec(), vc.getTVec(), fc->K, fc->distCoeff, reprojected_pt_set1);
    vc.addX(9);
    vc.rotZ(5);
    projectPoints(pt_3d, vc.getRVec(), vc.getTVec(), fc->K, fc->distCoeff, reprojected_pt_set2);


    Mat img1(480, 640, CV_8UC3, Scalar(0, 0, 0));
    for (int i = 0; i < reprojected_pt_set1.size(); i++) {
        circle(img1, reprojected_pt_set1[i], 1, Scalar(255, 255, 255), cv::FILLED);
    }

    imshow("img1", img1);

    Mat img2(480, 640, CV_8UC3, Scalar(0, 0, 0));
    for (int i = 0; i < reprojected_pt_set2.size(); i++) {
        circle(img2, reprojected_pt_set2[i], 1, Scalar(255, 255, 255), cv::FILLED);
    }

    imshow("img2", img2);

    char key = waitKey();

    OFReconstruction reconstruction(fc, reprojected_pt_set1, reprojected_pt_set2);
    reconstruction.reconstruct();


    return 0;
}