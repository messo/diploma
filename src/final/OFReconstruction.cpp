#include <iostream>
#include <opencv2/core/utility.hpp>
#include <opencv2/calib3d.hpp>
#include "OFReconstruction.h"
#include "Camera.hpp"
#include "Triangulation.h"

using namespace cv;
using namespace std;


OFReconstruction::OFReconstruction(cv::Ptr<Camera> cam, std::vector<cv::Point2f> pts1, std::vector<cv::Point2f> pts2)
        : cam(cam), pts1(pts1), pts2(pts2) {

}

void TakeSVDOfE(Mat_<double> &E, Mat &svd_u, Mat &svd_vt, Mat &svd_w) {
    //Using OpenCV's SVD
    SVD svd(E, SVD::MODIFY_A);
    svd_u = svd.u;
    svd_vt = svd.vt;
    svd_w = svd.w;

    // cout << "----------------------- SVD ------------------------\n";
    // cout << "U:\n" << svd_u << "\nW:\n" << svd_w << "\nVt:\n" << svd_vt << endl;
    // cout << "----------------------------------------------------\n";
}

bool DecomposeEtoRandT(
        Mat_<double> &E,
        Mat_<double> &R1,
        Mat_<double> &R2,
        Mat_<double> &t1,
        Mat_<double> &t2) {

    //Using HZ E decomposition
    Mat svd_u, svd_vt, svd_w;
    TakeSVDOfE(E, svd_u, svd_vt, svd_w);

    //check if first and second singular values are the same (as they should be)
    double singular_values_ratio = fabsf(svd_w.at<double>(0) / svd_w.at<double>(1));
    if (singular_values_ratio > 1.0) singular_values_ratio = 1.0 / singular_values_ratio; // flip ratio to keep it [0,1]
    if (singular_values_ratio < 0.7) {
        cout << "singular values are too far apart\n";
        return false;
    }

    Matx33d W(0, -1, 0,    //HZ 9.13
              1, 0, 0,
              0, 0, 1);
    Matx33d Wt(0, 1, 0,
               -1, 0, 0,
               0, 0, 1);
    R1 = svd_u * Mat(W) * svd_vt; //HZ 9.19
    R2 = svd_u * Mat(Wt) * svd_vt; //HZ 9.19
    t1 = svd_u.col(2); //u3
    t2 = -svd_u.col(2); //u3
    return true;
}

bool CheckCoherentRotation(cv::Mat_<double> &R) {
//    std::cout << "R; " << R << std::endl;

    if (fabs(determinant(R)) - 1.0 > 1e-07) {
        cerr << "det(R) != +-1.0, this is not a rotation matrix" << endl;
        return false;
    }

    return true;
}

bool TestTriangulation(const vector<CloudPoint> &pcloud, const Matx34d &P, vector<uchar> &status) {
    vector<Point3d> pcloud_pt3d = CloudPointsToPoints(pcloud);
    vector<Point3d> pcloud_pt3d_projected(pcloud_pt3d.size());

    Matx44d P4x4 = Matx44d::eye();
    for (int i = 0; i < 12; i++) P4x4.val[i] = P.val[i];

    perspectiveTransform(pcloud_pt3d, pcloud_pt3d_projected, P4x4);

    status.resize(pcloud.size(), 0);
    for (int i = 0; i < pcloud.size(); i++) {
        status[i] = (pcloud_pt3d_projected[i].z > 0) ? 1 : 0;
    }
    int count = countNonZero(status);

    double percentage = ((double) count / (double) pcloud.size());
    cout << count << "/" << pcloud.size() << " = " << percentage * 100.0 << "% are in front of camera" << endl;
    cout.flush();
    if (percentage < 0.75)
        return false; //less than 75% of the points are in front of the camera

    //check for coplanarity of points
    if (false) //not
    {
        cv::Mat_<double> cldm(pcloud.size(), 3);
        for (unsigned int i = 0; i < pcloud.size(); i++) {
            cldm.row(i)(0) = pcloud[i].pt.x;
            cldm.row(i)(1) = pcloud[i].pt.y;
            cldm.row(i)(2) = pcloud[i].pt.z;
        }
        cv::Mat_<double> mean;
        cv::PCA pca(cldm, mean, cv::PCA::DATA_AS_ROW);

        int num_inliers = 0;
        cv::Vec3d nrm = pca.eigenvectors.row(2);
        nrm = nrm / norm(nrm);
        cv::Vec3d x0 = pca.mean;
        double p_to_plane_thresh = sqrt(pca.eigenvalues.at<double>(2));

        for (int i = 0; i < pcloud.size(); i++) {
            Vec3d w = Vec3d(pcloud[i].pt) - x0;
            double D = fabs(nrm.dot(w));
            if (D < p_to_plane_thresh) num_inliers++;
        }

        cout << num_inliers << "/" << pcloud.size() << " are coplanar" << endl;
        if ((double) num_inliers / (double) (pcloud.size()) > 0.85)
            return false;
    }

    return true;
}

bool OFReconstruction::reconstruct() {
    vector<Point2f> pts1_good, pts2_good;

    vector<uchar> status(pts1.size());
    Mat F = findFundamentalMat(pts1, pts2, FM_RANSAC, 0.1, 0.99, status);

    cout << "Total: " << pts1.size() << endl;

    int count = 0;

    for (unsigned int i = 0; i < status.size(); i++) {
        if (status[i]) {
            pts1_good.push_back(pts1[i]);
            pts2_good.push_back(pts2[i]);
            count++;
        }
    }

    cout << "Good: " << count << endl;
    cout.flush();

    if (count < 5) {
        cout << "Not enough matches..." << endl;
        return false;
    }

    //Essential matrix: compute then extract cameras [R|t]
    Mat_<double> E = cam->K.t() * F * cam->K; //according to HZ (9.12)

    if (fabs(determinant(E)) > 1e-07) {
        cout << "det(E) != 0 : " << determinant(E) << endl;
        return false;
    }

    Mat_<double> R1(3, 3);
    Mat_<double> R2(3, 3);
    Mat_<double> t1(1, 3);
    Mat_<double> t2(1, 3);

    Matx34d P = cv::Matx34d(1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 1, 0);
    Matx34d P1 = cv::Matx34d(1, 0, 0, 50,
                             0, 1, 0, 0,
                             0, 0, 1, 0);

    resultingCloud.clear();

    //decompose E to P' , HZ (9.19)
    {
        if (!DecomposeEtoRandT(E, R1, R2, t1, t2)) return false;

        if (determinant(R1) + 1.0 < 1e-09) {
            //according to http://en.wikipedia.org/wiki/Essential_matrix#Showing_that_it_is_valid
            cout << "det(R) == -1 [" << determinant(R1) << "]: flip E's sign" << endl;
            E = -E;
            DecomposeEtoRandT(E, R1, R2, t1, t2);
        }
        if (!CheckCoherentRotation(R1)) {
            cout << "resulting rotation is not coherent\n";
            P1 = 0;
            return false;
        }

        P1 = Matx34d(R1(0, 0), R1(0, 1), R1(0, 2), t1(0),
                     R1(1, 0), R1(1, 1), R1(1, 2), t1(1),
                     R1(2, 0), R1(2, 1), R1(2, 2), t1(2));
//        cout << "Testing P1 " << endl << Mat(P1) << endl;

        vector<CloudPoint> pcloud, pcloud1;

        vector<Point> corresp;
        double reproj_error1 = TriangulatePoints(pts1_good, pts2_good, cam->K, cam->Kinv, cam->distCoeff, P, P1, pcloud,
                                                 corresp);
        double reproj_error2 = TriangulatePoints(pts2_good, pts1_good, cam->K, cam->Kinv, cam->distCoeff, P1, P,
                                                 pcloud1, corresp);
        vector<uchar> tmp_status;
        //check if pointa are triangulated --in front-- of cameras for all 4 ambiguations
        if (!TestTriangulation(pcloud, P1, tmp_status) || !TestTriangulation(pcloud1, P, tmp_status) ||
            reproj_error1 > 100.0 || reproj_error2 > 100.0) {
            P1 = Matx34d(R1(0, 0), R1(0, 1), R1(0, 2), t2(0),
                         R1(1, 0), R1(1, 1), R1(1, 2), t2(1),
                         R1(2, 0), R1(2, 1), R1(2, 2), t2(2));
//            cout << "Testing P1 " << endl << Mat(P1) << endl;

            pcloud.clear();
            pcloud1.clear();
            corresp.clear();
            reproj_error1 = TriangulatePoints(pts1_good, pts2_good, cam->K, cam->Kinv, cam->distCoeff, P, P1, pcloud,
                                              corresp);
            reproj_error2 = TriangulatePoints(pts2_good, pts1_good, cam->K, cam->Kinv, cam->distCoeff, P1, P, pcloud1,
                                              corresp);

            if (!TestTriangulation(pcloud, P1, tmp_status) || !TestTriangulation(pcloud1, P, tmp_status) ||
                reproj_error1 > 100.0 || reproj_error2 > 100.0) {
                if (!CheckCoherentRotation(R2)) {
                    cout << "resulting rotation is not coherent\n";
                    P1 = 0;
                    return false;
                }

                P1 = Matx34d(R2(0, 0), R2(0, 1), R2(0, 2), t1(0),
                             R2(1, 0), R2(1, 1), R2(1, 2), t1(1),
                             R2(2, 0), R2(2, 1), R2(2, 2), t1(2));
//                cout << "Testing P1 " << endl << Mat(P1) << endl;

                pcloud.clear();
                pcloud1.clear();
                corresp.clear();
                reproj_error1 = TriangulatePoints(pts1_good, pts2_good, cam->K, cam->Kinv, cam->distCoeff, P, P1,
                                                  pcloud, corresp);
                reproj_error2 = TriangulatePoints(pts2_good, pts1_good, cam->K, cam->Kinv, cam->distCoeff, P1, P,
                                                  pcloud1, corresp);

                if (!TestTriangulation(pcloud, P1, tmp_status) || !TestTriangulation(pcloud1, P, tmp_status) ||
                    reproj_error1 > 100.0 || reproj_error2 > 100.0) {
                    P1 = Matx34d(R2(0, 0), R2(0, 1), R2(0, 2), t2(0),
                                 R2(1, 0), R2(1, 1), R2(1, 2), t2(1),
                                 R2(2, 0), R2(2, 1), R2(2, 2), t2(2));
//                    cout << "Testing P1 " << endl << Mat(P1) << endl;

                    pcloud.clear();
                    pcloud1.clear();
                    corresp.clear();
                    reproj_error1 = TriangulatePoints(pts1_good, pts2_good, cam->K, cam->Kinv, cam->distCoeff, P, P1,
                                                      pcloud, corresp);
                    reproj_error2 = TriangulatePoints(pts1_good, pts2_good, cam->K, cam->Kinv, cam->distCoeff, P1, P,
                                                      pcloud1, corresp);

                    if (!TestTriangulation(pcloud, P1, tmp_status) || !TestTriangulation(pcloud1, P, tmp_status) ||
                        reproj_error1 > 100.0 || reproj_error2 > 100.0) {
                        cout << "Shit." << endl;
                        return false;
                    }
                }
            }
        }

        vector<double> depths;

        for (unsigned int i = 0; i < pcloud.size(); i++) {
            if (pcloud[i].reprojection_error < 10.0) {
                resultingCloud.push_back(pcloud[i]);
                depths.push_back(pcloud[i].pt.z);
            }
        }

        //show "range image"
        {
            double minVal, maxVal;
            minMaxLoc(depths, &minVal, &maxVal);
            Mat tmp(480, 640, CV_8UC3, Scalar(0, 0, 0));
            for (unsigned int i = 0; i < resultingCloud.size(); i++) {
                double _d = MAX(MIN((resultingCloud[i].pt.z - minVal) / (maxVal - minVal), 1.0), 0.0);
                circle(tmp, corresp[i], 1, Scalar(255 * (1.0 - (_d)), 255 * (1.0 - (_d)), 255 * (1.0 - (_d))),
                       cv::FILLED);
            }
            imshow("Depth Map", tmp);
        }
    }

    return true;
}
