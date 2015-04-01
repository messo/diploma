#include "Common.h"
#include "Camera.hpp"

#include <fstream>
#include <opencv2/core/mat.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;
using namespace std;

std::pair<int, int> makePair(const cv::Point &pt) {
    return std::make_pair(pt.x, pt.y);
}

vector<Point3d> CloudPointsToPoints(const vector<CloudPoint> &cpts) {
    vector<Point3d> out;
    for (unsigned int i = 0; i < cpts.size(); i++) {
        out.push_back(cpts[i].pt);
    }
    return out;
}

void writeCloudPoints(const vector<CloudPoint> &cpts) {
    writeCloudPoints("cloud.ply", cpts);
}

void writeCloudPoints(const string &fileName, const vector<CloudPoint> &cpts) {
    ofstream myfile;

    myfile.open("/media/balint/Data/Linux/cloud/" + fileName);
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

bool CheckCoherentRotation(cv::Mat_<double> &R) {
//    std::cout << "R; " << R << std::endl;

    if (fabs(determinant(R)) - 1.0 > 1e-07) {
        cerr << "det(R) != +-1.0, this is not a rotation matrix" << endl;
        return false;
    }

    return true;
}

bool FindPoseEstimation(
        cv::Ptr<Camera> camera,
        cv::Mat_<double> &rvec,
        cv::Mat_<double> &t,
        cv::Mat_<double> &R,
        std::vector<cv::Point3f> ppcloud,
        std::vector<cv::Point2f> imgPoints,
        std::vector<cv::Point2f> &reprojected
) {
    if (ppcloud.size() <= 7 || imgPoints.size() <= 7 || ppcloud.size() != imgPoints.size()) {
        //something went wrong aligning 3D to 2D points..
        cerr << "couldn't find [enough] corresponding cloud points... (only " << ppcloud.size() << ")" << endl;
        return false;
    }

    vector<int> inliers;
    if (true) {
        //use CPU
        double minVal, maxVal;
        cv::minMaxIdx(imgPoints, &minVal, &maxVal);
        cv::solvePnPRansac(ppcloud, imgPoints, camera->K, camera->distCoeff, rvec,
                           t, true, 1000, 8.0, 0.99, inliers, cv::SOLVEPNP_EPNP);
        //CV_PROFILE("solvePnP",cv::solvePnP(ppcloud, imgPoints, K, distortion_coeff, rvec, t, true, CV_EPNP);)
    } else {
#ifdef HAVE_OPENCV_GPU
		//use GPU ransac
		//make sure datatstructures are cv::gpu compatible
		cv::Mat ppcloud_m(ppcloud); ppcloud_m = ppcloud_m.t();
		cv::Mat imgPoints_m(imgPoints); imgPoints_m = imgPoints_m.t();
		cv::Mat rvec_,t_;

		cv::gpu::solvePnPRansac(ppcloud_m,imgPoints_m,K_32f,distcoeff_32f,rvec_,t_,false);

		rvec_.convertTo(rvec,CV_64FC1);
		t_.convertTo(t,CV_64FC1);
#endif
    }

    std::vector<Point2f> projected3D;
    cv::projectPoints(ppcloud, rvec, t, camera->K, camera->distCoeff, projected3D);

    if (inliers.size() == 0) { //get inliers
        for (int i = 0; i < projected3D.size(); i++) {
            if (norm(projected3D[i] - imgPoints[i]) < 10.0) {
                reprojected.push_back(projected3D[i]);
                inliers.push_back(i);
            }
        }
    }

#if 0
	//display reprojected points and matches
//	cv::Mat reprojected; imgs_orig[working_view].copyTo(reprojected);
//	for(int ppt=0;ppt<imgPoints.size();ppt++) {
//		cv::line(reprojected,imgPoints[ppt],projected3D[ppt],cv::Scalar(0,0,255),1);
//	}
//	for (int ppt=0; ppt<inliers.size(); ppt++) {
//		cv::line(reprojected,imgPoints[inliers[ppt]],projected3D[inliers[ppt]],cv::Scalar(0,0,255),1);
//	}
//	for(int ppt=0;ppt<imgPoints.size();ppt++) {
//		cv::circle(reprojected, imgPoints[ppt], 2, cv::Scalar(255,0,0), CV_FILLED);
//		cv::circle(reprojected, projected3D[ppt], 2, cv::Scalar(0,255,0), CV_FILLED);
//	}
//	for (int ppt=0; ppt<inliers.size(); ppt++) {
//		cv::circle(reprojected, imgPoints[inliers[ppt]], 2, cv::Scalar(255,255,0), CV_FILLED);
//	}
//	stringstream ss; ss << "inliers " << inliers.size() << " / " << projected3D.size();
//	putText(reprojected, ss.str(), cv::Point(5,20), CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0,255,255), 2);
//
//	cv::imshow("__tmp", reprojected);
//	cv::waitKey(0);
//	cv::destroyWindow("__tmp");
#endif
    //cv::Rodrigues(rvec, R);
    //visualizerShowCamera(R,t,0,255,0,0.1);

    if (inliers.size() < (double) (imgPoints.size()) / 5.0) {
        cerr << "not enough inliers to consider a good pose (" << inliers.size() << "/" << imgPoints.size() << ")" <<
        endl;
        return false;
    }

    if (cv::norm(t) > 200.0) {
        // this is bad...
        cerr << "estimated camera movement is too big, skip this camera\r\n";
        return false;
    }

    cv::Rodrigues(rvec, R);
    if (!CheckCoherentRotation(R)) {
        cerr << "rotation is incoherent. we should try a different base view..." << endl;
        return false;
    }

    std::cout << "found t = " << t << "\nR = \n" << R << std::endl;
    return true;
}