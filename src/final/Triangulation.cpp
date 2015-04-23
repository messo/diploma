#include "Triangulation.h"

#include <iostream>

using namespace std;
using namespace cv;


/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
Mat_<double> LinearLSTriangulation(Point3d u,        //homogenous image point (u,v,1)
                                   Matx34d P,        //camera 1 matrix
                                   Point3d u1,        //homogenous image point in 2nd camera
                                   Matx34d P1        //camera 2 matrix
) {

    //build matrix A for homogenous equation system Ax = 0
    //assume X = (x,y,z,1), for Linear-LS method
    //which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
    //	cout << "u " << u <<", u1 " << u1 << endl;
    //	Matx<double,6,4> A; //this is for the AX=0 case, and with linear dependence..
    //	A(0) = u.x*P(2)-P(0);
    //	A(1) = u.y*P(2)-P(1);
    //	A(2) = u.x*P(1)-u.y*P(0);
    //	A(3) = u1.x*P1(2)-P1(0);
    //	A(4) = u1.y*P1(2)-P1(1);
    //	A(5) = u1.x*P(1)-u1.y*P1(0);
    //	Matx43d A; //not working for some reason...
    //	A(0) = u.x*P(2)-P(0);
    //	A(1) = u.y*P(2)-P(1);
    //	A(2) = u1.x*P1(2)-P1(0);
    //	A(3) = u1.y*P1(2)-P1(1);
    Matx43d A(u.x * P(2, 0) - P(0, 0), u.x * P(2, 1) - P(0, 1), u.x * P(2, 2) - P(0, 2),
              u.y * P(2, 0) - P(1, 0), u.y * P(2, 1) - P(1, 1), u.y * P(2, 2) - P(1, 2),
              u1.x * P1(2, 0) - P1(0, 0), u1.x * P1(2, 1) - P1(0, 1), u1.x * P1(2, 2) - P1(0, 2),
              u1.y * P1(2, 0) - P1(1, 0), u1.y * P1(2, 1) - P1(1, 1), u1.y * P1(2, 2) - P1(1, 2)
    );
    Matx41d B(-(u.x * P(2, 3) - P(0, 3)),
              -(u.y * P(2, 3) - P(1, 3)),
              -(u1.x * P1(2, 3) - P1(0, 3)),
              -(u1.y * P1(2, 3) - P1(1, 3)));

    Mat_<double> X;
    solve(A, B, X, DECOMP_SVD);

    return X;
}


/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
Mat_<double> IterativeLinearLSTriangulation(Point3d u,    //homogenous image point (u,v,1)
                                            Matx34d P,            //camera 1 matrix
                                            Point3d u1,            //homogenous image point in 2nd camera
                                            Matx34d P1            //camera 2 matrix
) {
    double wi = 1, wi1 = 1;
    Mat_<double> X(4, 1);
    Mat_<double> X_ = LinearLSTriangulation(u, P, u1, P1);
    X(0) = X_(0);
    X(1) = X_(1);
    X(2) = X_(2);
    X(3) = 1.0;
    for (int i = 0; i < 10; i++) { //Hartley suggests 10 iterations at most
        //recalculate weights
        double p2x = Mat_<double>(Mat_<double>(P).row(2) * X)(0);
        double p2x1 = Mat_<double>(Mat_<double>(P1).row(2) * X)(0);

        //breaking point
        if (fabsf(wi - p2x) <= EPSILON && fabsf(wi1 - p2x1) <= EPSILON) break;

        wi = p2x;
        wi1 = p2x1;

        //reweight equations and solve
        Matx43d A((u.x * P(2, 0) - P(0, 0)) / wi, (u.x * P(2, 1) - P(0, 1)) / wi, (u.x * P(2, 2) - P(0, 2)) / wi,
                  (u.y * P(2, 0) - P(1, 0)) / wi, (u.y * P(2, 1) - P(1, 1)) / wi, (u.y * P(2, 2) - P(1, 2)) / wi,
                  (u1.x * P1(2, 0) - P1(0, 0)) / wi1, (u1.x * P1(2, 1) - P1(0, 1)) / wi1,
                  (u1.x * P1(2, 2) - P1(0, 2)) / wi1,
                  (u1.y * P1(2, 0) - P1(1, 0)) / wi1, (u1.y * P1(2, 1) - P1(1, 1)) / wi1,
                  (u1.y * P1(2, 2) - P1(1, 2)) / wi1
        );
        Mat_<double> B = (Mat_<double>(4, 1) << -(u.x * P(2, 3) - P(0, 3)) / wi,
                -(u.y * P(2, 3) - P(1, 3)) / wi,
                -(u1.x * P1(2, 3) - P1(0, 3)) / wi1,
                -(u1.y * P1(2, 3) - P1(1, 3)) / wi1
        );

        solve(A, B, X_, DECOMP_SVD);
        X(0) = X_(0);
        X(1) = X_(1);
        X(2) = X_(2);
        X(3) = 1.0;
    }
    return X;
}

//Triagulate points
double TriangulatePoints(long frameId, const vector<Point2f> &pt_set1,
                         const vector<Point2f> &pt_set2,
                         const Mat &K,
                         const Mat &Kinv,
                         const Mat &distcoeff,
                         const Matx34d &P,
                         const Matx34d &P1,
                         Cloud &pointcloud,
                         vector<Point> &correspImg1Pt) {

    correspImg1Pt.clear();

    Matx44d P1_(P1(0, 0), P1(0, 1), P1(0, 2), P1(0, 3),
                P1(1, 0), P1(1, 1), P1(1, 2), P1(1, 3),
                P1(2, 0), P1(2, 1), P1(2, 2), P1(2, 3),
                0, 0, 0, 1);
    Matx44d P1inv(P1_.inv());

    cout << "Triangulating...";
    double t = getTickCount();
    vector<double> reproj_error;
    unsigned int pts_size = pt_set1.size();

#if 0
    //Using OpenCV's triangulation
    //convert to Point2f

    //undistort
    Mat pt_set1_pt, pt_set2_pt;
    undistortPoints(pt_set1, pt_set1_pt, K, distcoeff);
    undistortPoints(pt_set2, pt_set2_pt, K, distcoeff);

    //triangulate
    Mat pt_set1_pt_2r = pt_set1_pt.reshape(1, 2);
    Mat pt_set2_pt_2r = pt_set2_pt.reshape(1, 2);
    Mat pt_3d_h(1, pts_size, CV_32FC4);
    cv::triangulatePoints(P, P1, pt_set1_pt_2r, pt_set2_pt_2r, pt_3d_h);

    //calculate reprojection
    vector<Point3f> pt_3d;
    convertPointsHomogeneous(pt_3d_h.reshape(4, 1), pt_3d);
    cv::Mat_<double> R = (cv::Mat_<double>(3, 3) << P(0, 0), P(0, 1), P(0, 2), P(1, 0), P(1, 1), P(1, 2), P(2, 0), P(2, 1), P(2, 2));
    Vec3d rvec;
    Rodrigues(R, rvec);
    Vec3d tvec(P(0, 3), P(1, 3), P(2, 3));
    vector<Point2f> reprojected_pt_set1;
    projectPoints(pt_3d, rvec, tvec, K, distcoeff, reprojected_pt_set1);

    for (unsigned int i = 0; i < pts_size; i++) {
        CloudPoint cp;
        cp.pt = pt_3d[i];
        pointcloud.push_back(cp);
        correspImg1Pt.push_back(pt_set1[i]);
        reproj_error.push_back(norm(pt_set1[i] - reprojected_pt_set1[i]));
    }

#else
    Mat_<double> KP1 = K * Mat(P1);
#pragma omp parallel for
    for (int i = 0; i < pts_size; i++) {
        Point2f kp = pt_set1[i];
        Point3d u(kp.x, kp.y, 1.0);
        Mat_<double> um = Kinv * Mat_<double>(u);
        u.x = um(0);
        u.y = um(1);
        u.z = um(2);

        Point2f kp1 = pt_set2[i];
        Point3d u1(kp1.x, kp1.y, 1.0);
        Mat_<double> um1 = Kinv * Mat_<double>(u1);
        u1.x = um1(0);
        u1.y = um1(1);
        u1.z = um1(2);

        Mat_<double> X = IterativeLinearLSTriangulation(u, P, u1, P1);

//		cout << "3D Point: " << X << endl;
//		Mat_<double> x = Mat(P1) * X;
//		cout <<	"P1 * Point: " << x << endl;
//		Mat_<double> xPt = (Mat_<double>(3,1) << x(0),x(1),x(2));
//		cout <<	"Point: " << xPt << endl;
        Mat_<double> xPt_img = KP1 * X;                //reproject
//		cout <<	"Point * K: " << xPt_img << endl;
        Point2f xPt_img_(xPt_img(0) / xPt_img(2), xPt_img(1) / xPt_img(2));

        double reprj_err = norm(xPt_img_ - kp1);
        CloudPoint cp;
        cp.pt = Point3d(X(0), X(1), X(2));
        cp.reprojection_error = reprj_err;

#pragma omp critical
        {
            reproj_error.push_back(reprj_err);
            pointcloud.insert(frameId, cp, Point2i((int) pt_set1[i].x, (int) pt_set1[i].y));
            correspImg1Pt.push_back(pt_set1[i]);
        }
    }
#endif

    Scalar mse = mean(reproj_error);
    t = ((double) getTickCount() - t) / getTickFrequency();
    cout << "Done. (" << pointcloud.size() << "points, " << t << "s, mean reproj err = " << mse[0] << ")" << endl;

    return mse[0];
}

double TriangulatePoints(const vector<Point2f> &pt_set1,
                         const Mat &K1,
                         const Mat &Kinv1,
                         const vector<Point2f> &pt_set2,
                         const Mat &K2,
                         const Mat &Kinv2,

                         const Matx34d &P,
                         const Matx34d &P1,
                         vector<CloudPoint> &pointcloud,
                         vector<Point> &correspImg1Pt) {

    pointcloud.clear();
    correspImg1Pt.clear();

    Matx44d P1_(P1(0, 0), P1(0, 1), P1(0, 2), P1(0, 3),
                P1(1, 0), P1(1, 1), P1(1, 2), P1(1, 3),
                P1(2, 0), P1(2, 1), P1(2, 2), P1(2, 3),
                0, 0, 0, 1);
    Matx44d P1inv(P1_.inv());

    cout << "Triangulating...";
    double t = getTickCount();
    vector<double> reproj_error;
    unsigned int pts_size = pt_set1.size();

    Mat_<double> KP1 = K2 * Mat(P1);
#pragma omp parallel for
    for (int i = 0; i < pts_size; i++) {
        Point2f kp = pt_set1[i];
        Point3d u(kp.x, kp.y, 1.0);
        Mat_<double> um = Kinv1 * Mat_<double>(u);
        u.x = um(0);
        u.y = um(1);
        u.z = um(2);

        Point2f kp1 = pt_set2[i];
        Point3d u1(kp1.x, kp1.y, 1.0);
        Mat_<double> um1 = Kinv2 * Mat_<double>(u1);
        u1.x = um1(0);
        u1.y = um1(1);
        u1.z = um1(2);

        Mat_<double> X = IterativeLinearLSTriangulation(u, P, u1, P1);

//		cout << "3D Point: " << X << endl;
//		Mat_<double> x = Mat(P1) * X;
//		cout <<	"P1 * Point: " << x << endl;
//		Mat_<double> xPt = (Mat_<double>(3,1) << x(0),x(1),x(2));
//		cout <<	"Point: " << xPt << endl;
        Mat_<double> xPt_img = KP1 * X;                //reproject
//		cout <<	"Point * K: " << xPt_img << endl;
        Point2f xPt_img_(xPt_img(0) / xPt_img(2), xPt_img(1) / xPt_img(2));

        double reprj_err = norm(xPt_img_ - kp1);
        CloudPoint cp;
        cp.pt = Point3d(X(0), X(1), X(2));
        cp.reprojection_error = reprj_err;

#pragma omp critical
        {
            reproj_error.push_back(reprj_err);
            pointcloud.push_back(cp);
            correspImg1Pt.push_back(pt_set1[i]);
        }
    }

    Scalar mse = mean(reproj_error);
    t = ((double) getTickCount() - t) / getTickFrequency();
    cout << "ROY Done. (" << pointcloud.size() << "points, " << t << "s, mean reproj err = " << mse[0] << ")" << endl;

    return mse[0];
}

bool TestTriangulation(const Cloud &pcloud, const Matx34d &P, vector<uchar> &status) {
    vector<Point3d> pcloud_pt3d = CloudPointsToPoints(pcloud.points);
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
//        cv::Mat_<double> cldm(pcloud.size(), 3);
//        for (unsigned int i = 0; i < pcloud.size(); i++) {
//            cldm.row(i)(0) = pcloud[i].pt.x;
//            cldm.row(i)(1) = pcloud[i].pt.y;
//            cldm.row(i)(2) = pcloud[i].pt.z;
//        }
//        cv::Mat_<double> mean;
//        cv::PCA pca(cldm, mean, cv::PCA::DATA_AS_ROW);
//
//        int num_inliers = 0;
//        cv::Vec3d nrm = pca.eigenvectors.row(2);
//        nrm = nrm / norm(nrm);
//        cv::Vec3d x0 = pca.mean;
//        double p_to_plane_thresh = sqrt(pca.eigenvalues.at<double>(2));
//
//        for (int i = 0; i < pcloud.size(); i++) {
//            Vec3d w = Vec3d(pcloud[i].pt) - x0;
//            double D = fabs(nrm.dot(w));
//            if (D < p_to_plane_thresh) num_inliers++;
//        }
//
//        cout << num_inliers << "/" << pcloud.size() << " are coplanar" << endl;
//        if ((double) num_inliers / (double) (pcloud.size()) > 0.85)
//            return false;
    }

    return true;
}

double cvTriangulatePoints(const vector<Point2f> &points1, const Ptr<Camera> &cam1, const CameraPose &pose1,
                           const vector<Point2f> &points2, const Ptr<Camera> &cam2, const CameraPose &pose2,
                           vector<CloudPoint> &pointcloud) {

    cout << "Triangulating...";
    double t = getTickCount();

    int count = (int) points1.size();

    if(count == 0) {
        cout << "no points." << endl;
        return -1.0;
    }

    Mat normalized1, normalized2;

    undistortPoints(points1, normalized1, cam1->cameraMatrix, cam1->distCoeffs);
    undistortPoints(points2, normalized2, cam2->cameraMatrix, cam2->distCoeffs);

    cv::Mat points3D_h(4, count, CV_32FC1);
    cv::triangulatePoints(Mat(pose1.getRT()), Mat(pose2.getRT()), normalized1, normalized2, points3D_h);
    cv::Mat points3D;
    convertPointsFromHomogeneous(Mat(points3D_h.t()).reshape(4, 1), points3D);

    // reprojection

    Mat reprojected1, reprojected2;
    projectPoints(points3D, pose1.rvec, pose1.tvec, cam1->cameraMatrix, cam1->distCoeffs, reprojected1);
    projectPoints(points3D, pose2.rvec, pose2.tvec, cam2->cameraMatrix, cam2->distCoeffs, reprojected2);

    vector<double> reproj_error;
    for (int i = 0; i < count; i++) {
        double reprj_err1 = norm(reprojected1.at<Point2f>(i) - points1[i]);
        double reprj_err2 = norm(reprojected2.at<Point2f>(i) - points2[i]);

        CloudPoint cp;
        const Point3f &pt3f = points3D.at<Point3f>(i);
        cp.pt = Point3d(pt3f.x, pt3f.y, pt3f.z);
        cp.reprojection_error = max(reprj_err1, reprj_err2);

        reproj_error.push_back(cp.reprojection_error);
        pointcloud.push_back(cp);
    }

    Scalar mse = mean(reproj_error);
    t = ((double) getTickCount() - t) / getTickFrequency();
    cout << "CV Done. (" << pointcloud.size() << "points, " << t << "s, mean reproj err = " << mse[0] << ")" << endl;

    return mse[0];
}