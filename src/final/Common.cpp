#include "camera/Camera.hpp"
#include "camera/CameraPose.h"
#include "camera/RealCamera.hpp"
#include "Common.h"
#include "PerformanceMonitor.h"

#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Geometry>

using namespace cv;
using namespace std;

//Point2f LEFT_SHIFT(240, 120);
//Point2f RIGHT_SHIFT(330, 120);
//Size SIZE(160, 240);

// 320x240 középső
//Point2f LEFT_SHIFT(160, 120);
//Point2f RIGHT_SHIFT(250, 120);
//Size SIZE(320, 240);
float RATIO = 1.0f;

// 160x120 középső
//Point2f LEFT_SHIFT(260, 120);
//Point2f RIGHT_SHIFT(186, 120);
//Size SIZE(160, 140);

Point2f LEFT_SHIFT(0, 0);
Point2f RIGHT_SHIFT(0, 0);
Size SIZE(640, 480);
//
//float RATIO = 2.0f;

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
    Mat result = Mat::zeros(max(left.rows, right.rows), left.cols + right.cols, left.type());

    left.copyTo(result(Rect(0, 0, left.cols, left.rows)));
    right.copyTo(result(Rect(left.cols, 0, right.cols, right.rows)));

    return result;
}

Mat mergeImagesVertically(const Mat &left, const Mat &right) {
    Mat result = Mat::zeros(left.rows + right.rows, max(left.cols, right.cols), left.type());

    left.copyTo(result(Rect(0, 0, left.cols, left.rows)));
    right.copyTo(result(Rect(0, left.rows, right.cols, right.rows)));

    return result;
}

void shiftImage(const cv::Mat &input, const cv::Rect &boundingRect,
                const cv::Point2i &translation, cv::Mat &output) {
    output.setTo(0);
    Rect newBoundingRect(boundingRect + translation);

    if ((newBoundingRect & Rect(Point(0, 0), output.size())) == newBoundingRect) {
        input(boundingRect).copyTo(output(newBoundingRect));
    } else {
        newBoundingRect &= Rect(Point(0, 0), output.size());
        Rect _boundingRect(newBoundingRect - translation);

        input(_boundingRect).copyTo(output(newBoundingRect));
    }
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
        cv::solvePnPRansac(ppcloud, imgPoints, camera->cameraMatrix, camera->distCoeffs, rvec,
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
    cv::projectPoints(ppcloud, rvec, t, camera->cameraMatrix, camera->distCoeffs, projected3D);

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

void drawGridXY(cv::Mat &img, cv::Ptr<Camera> camera, cv::Ptr<CameraPose> cameraPose) {
    int minX = -2;
    int maxX = 4;
    int minY = -2;
    int maxY = 2;
    int maxZ = 0;
    int step = 8;

    std::vector<Point3f> gridPoints;
    for (int z = 0; z <= maxZ; z++) {
        for (int x = minX; x <= maxX; x++) {
            gridPoints.push_back(Point3f(x * step, (minY) * step, -z * step * 2));
            gridPoints.push_back(Point3f(x * step, (maxY) * step, -z * step * 2));
        }
        for (int y = minY; y <= maxY; y++) {
            gridPoints.push_back(Point3f((minX) * step, y * step, -z * step * 2));
            gridPoints.push_back(Point3f((maxX) * step, y * step, -z * step * 2));
        }
    }
    gridPoints.push_back(Point3f(0, 0, 0));
    gridPoints.push_back(Point3f(step, 0, 0));
    gridPoints.push_back(Point3f(0, step, 0));

    std::vector<Point2f> gridImagePoints;
    projectPoints(gridPoints, cameraPose->rvec, cameraPose->tvec, camera->cameraMatrix, camera->distCoeffs,
                  gridImagePoints);

    Rect imageRect(0, 0, 640, 480);
    int i = 0;
    RNG rng;
    Scalar color(0, 0, 0);
    for (int z = 0; z <= maxZ; z++) {
        int icolor = (unsigned) rng;
        for (int x = minX; x <= maxX; x++) {
            if (gridImagePoints[i].inside(imageRect) && gridImagePoints[i + 1].inside(imageRect)) {
                line(img, gridImagePoints[i], gridImagePoints[i + 1], color, 2, LINE_AA);
            }
            i += 2;
        }
        for (int y = minY; y <= maxY; y++) {
            if (gridImagePoints[i].inside(imageRect) && gridImagePoints[i + 1].inside(imageRect)) {
                line(img, gridImagePoints[i], gridImagePoints[i + 1], color, 2, LINE_AA);
            }
            i += 2;
        }
    }

    Point2f offset(2, -6);

    circle(img, gridImagePoints[gridImagePoints.size() - 3], 2, Scalar(0, 0, 255), 2, LINE_AA);
    //putText(img, "(0,0,0)", gridImagePoints[gridImagePoints.size()-3] + Point2f(-100, -6), FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 0, 0), 2, LINE_AA);
    putText(img, "(0,0,0)", gridImagePoints[gridImagePoints.size() - 3] + Point2f(-100, -6), FONT_HERSHEY_PLAIN, 2.0,
            Scalar(255, 255, 255), 2, LINE_AA);

    circle(img, gridImagePoints[gridImagePoints.size() - 2], 2, Scalar(0, 0, 255), 2, LINE_AA);
    putText(img, "(8,0,0)", gridImagePoints[gridImagePoints.size() - 2] + offset, FONT_HERSHEY_PLAIN, 2.0,
            Scalar(255, 255, 255), 2, LINE_AA);
    //putText(img, "(8,0,0)", gridImagePoints[gridImagePoints.size()-2] + offset, FONT_HERSHEY_PLAIN, 2.0, Scalar(255, 255, 255), 1, LINE_AA);

    circle(img, gridImagePoints[gridImagePoints.size() - 1], 2, Scalar(0, 0, 255), 2, LINE_AA);
    putText(img, "(0,8,0)", gridImagePoints[gridImagePoints.size() - 1] + offset, FONT_HERSHEY_PLAIN, 2.0,
            Scalar(255, 255, 255), 2, LINE_AA);
    //putText(img, "(0,8,0)", gridImagePoints[gridImagePoints.size()-1] + offset, FONT_HERSHEY_PLAIN, 2.0, Scalar(255, 255, 255), 1, LINE_AA);
}

void drawBoxOnChessboard(Mat inputImage, Ptr<Camera> camera, Ptr<CameraPose> pose) {
    // coordinates for box
    vector<Point3f> objectPoints;
    objectPoints.push_back(Point3f(0, 0, 0));
    objectPoints.push_back(Point3f(0, 5, 0));
    objectPoints.push_back(Point3f(8, 5, 0));
    objectPoints.push_back(Point3f(8, 0, 0));

    objectPoints.push_back(Point3f(0, 0, -5));
    objectPoints.push_back(Point3f(0, 5, -5));
    objectPoints.push_back(Point3f(8, 5, -5));
    objectPoints.push_back(Point3f(8, 0, -5));

    // calculating imagePoints
    vector<Point2f> imagePoints;
    projectPoints(objectPoints, pose->rvec, pose->tvec, camera->cameraMatrix, camera->distCoeffs, imagePoints);

    // drawing
    line(inputImage, imagePoints[0], imagePoints[1], Scalar(0, 0, 255), 1, LINE_AA);
    line(inputImage, imagePoints[1], imagePoints[2], Scalar(0, 0, 255), 1, LINE_AA);
    line(inputImage, imagePoints[2], imagePoints[3], Scalar(0, 0, 255), 1, LINE_AA);
    line(inputImage, imagePoints[3], imagePoints[0], Scalar(0, 0, 255), 1, LINE_AA);

    line(inputImage, imagePoints[4], imagePoints[5], Scalar(0, 0, 255), 1, LINE_AA);
    line(inputImage, imagePoints[5], imagePoints[6], Scalar(0, 0, 255), 1, LINE_AA);
    line(inputImage, imagePoints[6], imagePoints[7], Scalar(0, 0, 255), 1, LINE_AA);
    line(inputImage, imagePoints[7], imagePoints[4], Scalar(0, 0, 255), 1, LINE_AA);

    line(inputImage, imagePoints[0], imagePoints[4], Scalar(0, 0, 255), 1, LINE_AA);
    line(inputImage, imagePoints[1], imagePoints[5], Scalar(0, 0, 255), 1, LINE_AA);
    line(inputImage, imagePoints[2], imagePoints[6], Scalar(0, 0, 255), 1, LINE_AA);
    line(inputImage, imagePoints[3], imagePoints[7], Scalar(0, 0, 255), 1, LINE_AA);
}

Point moveToTheCenter(Mat image, Mat mask) {
    Rect boundingRect(cv::boundingRect(mask));

    Point translation((320 - boundingRect.width / 2) - boundingRect.x,
                      (240 - boundingRect.height / 2) - boundingRect.y);

    // translate the image
    Mat image_ = image.clone();
    shiftImage(image_, boundingRect, translation, image);
    // translate the mask
    Mat mask_ = mask.clone();
    shiftImage(mask_, boundingRect, translation, mask);

    return translation;
}

std::vector<std::vector<Mat>> getFramesFromCameras(std::vector<Ptr<Camera>> &cameras,
                                                   std::vector<Ptr<ForegroundMaskCalculator>> &maskCalculators) {

    PerformanceMonitor::get()->maskCalculationStarted();

    std::vector<std::vector<Mat>> result(2);
    result[0].resize(2);
    result[1].resize(2);

#pragma omp parallel for
    for (int i = 0; i < 2; i++) {
        // std::cout << "CAP THREAD: " << omp_get_thread_num() << std::endl;
        Mat image;
        cameras[i]->read(image); // readUndistorted

        Mat smaller_image;
        if (RATIO == 1.0f) {
            if (i == 0) {
                image(Rect(LEFT_SHIFT, SIZE)).copyTo(smaller_image);
            } else {
                image(Rect(RIGHT_SHIFT, SIZE)).copyTo(smaller_image);
            }
        } else {
            resize(image, smaller_image, Size(), 1 / RATIO, 1 / RATIO, INTER_AREA);
        }

        result[1][i] = maskCalculators[i]->calculate(smaller_image);

//        Mat gray;
//        cvtColor(image, gray, COLOR_BGR2GRAY);
//        equalizeHist(gray, result[0][i]);

        smaller_image.copyTo(result[0][i]);

//        if (i == 0) {
//            imshow("image", image);
//            imwrite("/media/balint/Data/cucc/image" + std::to_string(counter) + ".png", image);
//            imshow("mask", mask);
//            imwrite("/media/balint/Data/cucc/mask" + std::to_string(counter) + ".png", mask);
//            counter++;
//        }

        //drawGridXY(leftImage, leftCamera, leftCameraPose);
    }

    PerformanceMonitor::get()->maskCalculationFinished();

    return result;
}

Point2f magicVector(const std::vector<Point2f> &vector) {
    Mat_<double> A(vector.size() * 2, 2);
    Mat_<double> B(vector.size() * 2, 1);

    for (int i = 0; i < vector.size(); i++) {
        A(2 * i, 0) = 1;
        A(2 * i, 1) = 0;
        A(2 * i + 1, 0) = 0;
        A(2 * i + 1, 1) = 1;

        B(i * 2, 0) = vector[i].x;
        B(i * 2 + 1, 0) = vector[i].y;
    }

    Mat_<double> X(2, 1);
    solve(A, B, X, DECOMP_SVD);
    return Point2f(X(0), X(1));
}

Mat slerp(Mat rvec1, Mat rvec2, double ratio) {
    Mat R1, R2;
    Rodrigues(rvec1, R1);
    Rodrigues(rvec2, R2);

    Eigen::Matrix3d r1;
    Eigen::Matrix3d r2;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            r1(i, j) = R1.at<double>(i, j);
            r2(i, j) = R2.at<double>(i, j);
        }
    }

    Eigen::Quaterniond q1(r1);
    Eigen::Quaterniond q2(r2);
    Eigen::Matrix3d R = q1.slerp(ratio, q2).toRotationMatrix();

    Mat rot(3, 3, CV_64FC1);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            rot.at<double>(i, j) = R(i, j);
        }
    }

    Mat rvec;
    Rodrigues(rot, rvec);

    return rvec;
}
