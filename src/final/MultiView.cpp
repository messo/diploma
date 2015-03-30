#include "MultiView.h"
#include "Triangulation.h"

using namespace std;
using namespace cv;

MultiView::MultiView(cv::Ptr<Camera> camera) : cam(camera) {

}

void MultiView::addP(long frameId, cv::Matx34d P) {
    Pmats[frameId] = P;
}

void MultiView::reconstructNext(long frameId1, const vector<Point2f> &points1,
                                long frameId2, const vector<Point2f> &points2) {

    // cout << "frameId1: " << frameId1 << endl;

    map<pair<int, int>, int> &a = cloud.lookupIdxBy2D[frameId1];

    vector<Point3f> points3d;
    vector<Point2f> imgpoints;

    // collecting 2D matching points for 3D coordinates

    for (int i = 0; i < points1.size(); i++) {
        Point2i pt(points1[i]);
        map<pair<int, int>, int>::iterator it(a.find(makePair(pt)));
        if (it != a.end()) {
            // we have a cloudpoint for points1[i], let's save the 2d point
            imgpoints.push_back(points2[i]);
            // and the cloudpoint
            points3d.push_back(cloud.points[it->second].pt);

            // cout << pt << ": " << a.count(makePair(pt)) << endl;
        }
    }

    //cout.flush();

    Mat_<double> rvec(1, 3);
    Mat_<double> t(1, 3);
    Mat_<double> R(3, 3);

    vector<Point2f> reprojected;
    FindPoseEstimation(cam, rvec, t, R, points3d, imgpoints, reprojected);

    Matx34d P2 = Matx34d(R(0, 0), R(0, 1), R(0, 2), t(0),
                         R(1, 0), R(1, 1), R(1, 2), t(1),
                         R(2, 0), R(2, 1), R(2, 2), t(2));
    this->addP(frameId2, P2);

    cout << "===========" << endl;
    cout << Pmats[frameId1] << endl;
    cout << Pmats[frameId2] << endl;

    // triangulate new points

    Cloud pc;
    vector<Point> correspond;
    double reproj_error = TriangulatePoints(frameId1, points1, points2, cam->K, cam->Kinv, cam->distCoeff,
                                            Pmats[frameId1], Pmats[frameId2],
                                            pc, correspond);
    // std::cout << "triangulation reproj error " << reproj_error << std::endl;

    vector<uchar> trig_status;
    if (!TestTriangulation(pc, Pmats[frameId1], trig_status) || !TestTriangulation(pc, Pmats[frameId2], trig_status)) {
        cerr << "Triangulation did not succeed" << endl;
        return;
    }


    for (unsigned int i = 0; i < pc.size(); i++) {
        // FIXME -- what this should be??? 80% percentile???
        // SAME AS IN OF RECONSTRUCTION,, move this out??

        if (pc.points[i].reprojection_error < 10.0) {
            cloud.insert(pc.points[i], frameId1, points1[i], frameId2, points2[i]);
        }
    }

    writeCloudPoints("cloud1.ply", cloud.points);
}
