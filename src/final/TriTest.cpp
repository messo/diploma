#include <iostream>
#include <opencv2/core/types.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "camera/VirtualCamera.h"
#include "Triangulation.h"
#include "optical_flow/OFReconstruction.h"
#include "camera/FakeCamera.h"

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
    projectPoints(pt_3d, vc.getRVec(), vc.getTVec(), fc->cameraMatrix, fc->distCoeffs, reprojected_pt_set1);
    vc.addX(9);
    vc.rotZ(5);
    projectPoints(pt_3d, vc.getRVec(), vc.getTVec(), fc->cameraMatrix, fc->distCoeffs, reprojected_pt_set2);


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

    OFReconstruction reconstruction(fc, 1, reprojected_pt_set1, 2, reprojected_pt_set2);
    reconstruction.reconstruct();

    writeCloudPoints("cloud1.ply", reconstruction.resultingCloud.points);

    // CV:TRIANGUALTE

    cv::Mat points4D(1, reconstruction.resultingCloud.size(), CV_32FC4);
    cv::triangulatePoints(fc->cameraMatrix * Mat(reconstruction.P1),
                          fc->cameraMatrix * Mat(reconstruction.P2),
                          Mat(reprojected_pt_set1).reshape(2, 1),
                          Mat(reprojected_pt_set2).reshape(2, 1),
                          points4D);

    //std::cout << points4D;
    //std::cout.flush();

    cv::Mat points3D(1, reconstruction.resultingCloud.size(), CV_32FC3);
    convertPointsFromHomogeneous(points4D.reshape(4, 1), points3D);

    vector<CloudPoint> cloud2;
    for(int i=0; i<reconstruction.resultingCloud.size(); i++) {
        Vec3f v = points3D.at<Vec3f>(i);

        CloudPoint cp;
        cp.pt = Point3d(v[0], v[1], v[2]);
        cloud2.push_back(cp);
    }

    writeCloudPoints("cloud2.ply", cloud2);

    return 0;
}