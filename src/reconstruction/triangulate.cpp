#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching/warpers.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

// #include <opencv2/core/opengl_interop.hpp>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cwchar>
#include <string>

#include "Magic.h"
#include "Canvas.h"
#include "OpenGLRenderer.h"
#include "PCDWriter.h"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

void writePLY4D(const Mat& xyzw, const std::vector<Point2f> imgPoints, const Mat& img) {
	FILE* fp = fopen("magic4.ply", "wt");
	fprintf(fp, "%s", "ply\n");
	fprintf(fp, "%s", "format ascii 1.0\n");

	int count = 0;

	std::vector<Vec4d> points;
	std::vector<Vec3b> colors;
	for (int i = 0; i < xyzw.cols; i++) {
		Vec4d point = xyzw.col(i);
		Vec4d point3 = point / point.val[3];

		if (point3.val[2] > 0.0f) {
			points.push_back(point3);
			colors.push_back(img.at<Vec3b>(imgPoints[i]));
			// std::cout << point3.val[2] << std::endl;
		}
	}

	double avg = 0.0;
	for (int i = 0; i < points.size(); i++) {
		avg += points[i].val[2];
	}

	avg /= points.size();
	std::cout << avg << std::endl;

	std::vector<Vec4d> good_points;
	std::vector<Vec3b> good_colors;
	for (int i = 0; i < points.size(); i++) {
		if (abs(points[i].val[2] - avg) < avg / 4) {
			good_points.push_back(points[i]);
			good_colors.push_back(colors[i]);
		}
	}

	fprintf(fp, "element vertex %d\n", good_points.size());
	fprintf(fp, "%s", "property float x\n");
	fprintf(fp, "%s", "property float y\n");
	fprintf(fp, "%s", "property float z\n");
	fprintf(fp, "%s", "property uchar blue\n");
	fprintf(fp, "%s", "property uchar green\n");
	fprintf(fp, "%s", "property uchar red\n");
	fprintf(fp, "%s", "end_header\n");

	for (int i = 0; i < good_points.size(); i++) {
		fprintf(fp, "%f %f %f %d %d %d\n", good_points[i].val[0]/10, good_points[i].val[1]/10, good_points[i].val[2]/10,
				good_colors[i].val[0], good_colors[i].val[1], good_colors[i].val[2]);
	}

	fclose(fp);
}

void writePLY3D(const Mat& xyz, const std::vector<Point2f> imgPoints, const Mat& img) {
	FILE* fp = fopen("magic3.ply", "wt");
	fprintf(fp, "%s", "ply\n");
	fprintf(fp, "%s", "format ascii 1.0\n");

	int count = 0;

	fprintf(fp, "element vertex %d\n", xyz.rows * xyz.cols);
	fprintf(fp, "%s", "property float x\n");
	fprintf(fp, "%s", "property float y\n");
	fprintf(fp, "%s", "property float z\n");
	fprintf(fp, "%s", "property uchar blue\n");
	fprintf(fp, "%s", "property uchar green\n");
	fprintf(fp, "%s", "property uchar red\n");
	fprintf(fp, "%s", "end_header\n");

	std::vector<Vec3d> points;
	std::vector<Vec3b> colors;
	for (int i = 0; i < xyz.rows; i++) {
		Vec3f point3 = xyz.at<Vec3f>(i, 0);

		if (point3.val[2] > 0.0f) {
			points.push_back(point3);
			colors.push_back(img.at<Vec3b>(imgPoints[i]));
			// std::cout << point3.val[2] << std::endl;
		}
	}

	for (int i = 0; i < points.size(); i++) {
		fprintf(fp, "%f %f %f %d %d %d\n", points[i][0], points[i][1], points[i][2], colors[i].val[0], colors[i].val[1],
				colors[i].val[2]);
	}

	fclose(fp);
}

void writePLY3D(const vector<Vec4d>& xyz0, const std::vector<Point2f> imgPoints, const Mat& img) {
	FILE* fp = fopen("magic3.ply", "wt");
	fprintf(fp, "%s", "ply\n");
	fprintf(fp, "%s", "format ascii 1.0\n");

	int count = 0;

	fprintf(fp, "element vertex %d\n", xyz0.size());
	fprintf(fp, "%s", "property float x\n");
	fprintf(fp, "%s", "property float y\n");
	fprintf(fp, "%s", "property float z\n");
	fprintf(fp, "%s", "property uchar blue\n");
	fprintf(fp, "%s", "property uchar green\n");
	fprintf(fp, "%s", "property uchar red\n");
	fprintf(fp, "%s", "end_header\n");

	for (size_t i = 0; i < xyz0.size(); i++) {
		Vec3b color = img.at<Vec3b>(imgPoints[i]);
		fprintf(fp, "%f %f %f %d %d %d\n", xyz0[i][0], xyz0[i][1], xyz0[i][2], color[0], color[1],
				color[2]);
	}

	fclose(fp);
}

Mat detectEdges(const string& name, const Mat& mask) {
	Mat src = imread(name);
	Mat src_gray, dst, dst_norm, leftEdges;
	cvtColor(src, src_gray, COLOR_BGR2GRAY);

	vector<Point2f> corners;
	int maxCorners = 100;
	double qualityLevel = 0.01;
	double minDistance = 10;
	int blockSize = 3;
	bool useHarrisDetector = false;
	double k = 0.04;

	goodFeaturesToTrack(src_gray, corners, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector,
			k);

	int r = 4;
	for (int i = 0; i < corners.size(); i++) {
		circle(src, corners[i], r, Scalar(255, 0, 0), -1, 8, 0);
	}

	return src;
}

/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
Mat_<double> LinearLSTriangulation(Point3d u,       //homogenous image point (u,v,1)
		Matx34d P,       //camera 1 matrix
		Point3d u1,      //homogenous image point in 2nd camera
		Matx34d P1       //camera 2 matrix
		) {
	//build matrix A for homogenous equation system Ax = 0
	//assume X = (x,y,z,1), for Linear-LS method
	//which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
	Matx43d A(u.x * P(2, 0) - P(0, 0), u.x * P(2, 1) - P(0, 1), u.x * P(2, 2) - P(0, 2), u.y * P(2, 0) - P(1, 0),
			u.y * P(2, 1) - P(1, 1), u.y * P(2, 2) - P(1, 2), u1.x * P1(2, 0) - P1(0, 0), u1.x * P1(2, 1) - P1(0, 1),
			u1.x * P1(2, 2) - P1(0, 2), u1.y * P1(2, 0) - P1(1, 0), u1.y * P1(2, 1) - P1(1, 1),
			u1.y * P1(2, 2) - P1(1, 2));
	Mat_<double> B = (Mat_<double>(4, 1) << -(u.x * P(2, 3) - P(0, 3)), -(u.y * P(2, 3) - P(1, 3)), -(u1.x * P1(2, 3) - P1(0, 3)), -(u1.y
			* P1(2, 3) - P1(1, 3)));

	Mat_<double> X;
	solve(A, B, X, DECOMP_SVD);

	return X;
}

const float EPSILON = 0.1f;

/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
Mat_<double> IterativeLinearLSTriangulation(Point3d u,    //homogenous image point (u,v,1)
		Matx34d P,          //camera 1 matrix
		Point3d u1,         //homogenous image point in 2nd camera
		Matx34d P1          //camera 2 matrix
		) {
	double wi = 1, wi1 = 1;
	Mat_<double> X(4, 1);
	for (int i = 0; i < 10; i++) { //Hartley suggests 10 iterations at most
		Mat_<double> X_ = LinearLSTriangulation(u, P, u1, P1);
		X(0) = X_(0);
		X(1) = X_(1);
		X(2) = X_(2);
		X_(3) = 1.0;

		//recalculate weights
		double p2x = Mat_<double>(Mat_<double>(P).row(2) * X)(0);
		double p2x1 = Mat_<double>(Mat_<double>(P1).row(2) * X)(0);

		//breaking point
		if (fabsf(wi - p2x) <= EPSILON && fabsf(wi1 - p2x1) <= EPSILON)
			break;

		wi = p2x;
		wi1 = p2x1;

		//reweight equations and solve
		Matx43d A((u.x * P(2, 0) - P(0, 0)) / wi, (u.x * P(2, 1) - P(0, 1)) / wi, (u.x * P(2, 2) - P(0, 2)) / wi,
				(u.y * P(2, 0) - P(1, 0)) / wi, (u.y * P(2, 1) - P(1, 1)) / wi, (u.y * P(2, 2) - P(1, 2)) / wi,
				(u1.x * P1(2, 0) - P1(0, 0)) / wi1, (u1.x * P1(2, 1) - P1(0, 1)) / wi1,
				(u1.x * P1(2, 2) - P1(0, 2)) / wi1, (u1.y * P1(2, 0) - P1(1, 0)) / wi1,
				(u1.y * P1(2, 1) - P1(1, 1)) / wi1, (u1.y * P1(2, 2) - P1(1, 2)) / wi1);
		Mat_<double> B = (Mat_<double>(4, 1) << -(u.x * P(2, 3) - P(0, 3)) / wi, -(u.y * P(2, 3) - P(1, 3)) / wi, -(u1.x
				* P1(2, 3) - P1(0, 3)) / wi1, -(u1.y * P1(2, 3) - P1(1, 3)) / wi1);

		solve(A, B, X_, DECOMP_SVD);
		X(0) = X_(0);
		X(1) = X_(1);
		X(2) = X_(2);
		X_(3) = 1.0;
	}
	return X;
}

int main(int argc, char** argv) {
	Magic magic;
	Canvas canvas(Size(640, 480), 3);

	Camera camera;
	camera.setTranslation(0.0, 0.0, 0.5);

	int picIdx = 0;
	bool drawEpi = false;
	int sc_type = 0;

	int count = 0;

//	OpenGLRenderer renderer(640, 480);
//	renderer.init();

	Mat mask_left = Mat::ones(Size(640, 480), CV_8U);
	Mat roi_left(mask_left, magic.validRoiLeft);
	roi_left = Scalar(255, 255, 255);

	Mat mask_right = Mat::ones(Size(640, 480), CV_8U);
	Mat roi_right(mask_right, magic.validRoiRight);
	roi_right = Scalar(255, 255, 255);

	Mat imgLeft = imread("left01_bal.jpg");
	Mat imgRight = imread("right01_bal.jpg");

	//Mat imgLeftRM = magic.readAndRemap("left01_bal.jpg", LEFT);
	//Mat imgRightRM = magic.readAndRemap("right01_bal.jpg", RIGHT);

	//-- Step 1: Detect the keypoints using SURF Detector
	double minHessian = 400;

	SURF surf(minHessian);
	SIFT sift(0, 3, 0.04, 10, 0.5);

	std::vector<KeyPoint> keypoints_left, keypoints_right;
	Mat descriptors_left, descriptors_right;

	surf(imgLeft, mask_left, keypoints_left, descriptors_left);
	surf(imgRight, mask_right, keypoints_right, descriptors_right);

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	std::vector<DMatch> matches;
	matcher.match(descriptors_left, descriptors_right, matches);

	double max_dist = 0;
	double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_left.rows; i++) {
		double dist = matches[i].distance;
		if (dist < min_dist)
			min_dist = dist;
		if (dist > max_dist)
			max_dist = dist;
	}

	std::vector<int> pointIndexesLeft;
	std::vector<int> pointIndexesRight;
	for (std::vector<cv::DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it) {
		// Get the indexes of the selected matchedkeypoints
		if (it->distance <= max(2 * min_dist, 0.02)) {
			pointIndexesLeft.push_back(it->queryIdx);
			pointIndexesRight.push_back(it->trainIdx);
		}
	}

	std::vector<cv::Point2f> selPointsLeft, selPointsRight;
	cv::KeyPoint::convert(keypoints_left, selPointsLeft, pointIndexesLeft);
	cv::KeyPoint::convert(keypoints_right, selPointsRight, pointIndexesRight);

	Mat leftMat(Size(selPointsLeft.size(), 1), CV_32FC2), rightMat(Size(selPointsLeft.size(), 1), CV_32FC2);
	for (size_t i = 0; i < selPointsLeft.size(); i++) {
		Vec2f& point = leftMat.at<Vec2f>(0, i);
		point.val[0] = selPointsLeft[i].x;
		point.val[1] = selPointsLeft[i].y;

		Vec2f& point2 = rightMat.at<Vec2f>(0, i);
		point2.val[0] = selPointsRight[i].x;
		point2.val[1] = selPointsRight[i].y;
	}

	//Mat np1, np2;
	//correctMatches(magic.F, leftMat, rightMat, np1, np2);

	std::vector<cv::Point2f> projPointsLeft, projPointsRight;
	for (size_t i = 0; i < selPointsLeft.size(); i++) {
		//if (abs(selPointsLeft[i].y - selPointsRight[i].y) < 3) {
			projPointsLeft.push_back(selPointsLeft[i]);
			projPointsRight.push_back(selPointsRight[i]);
			circle(imgLeft, selPointsLeft[i], 3, cv::Scalar(255, 255, 255), 2);
			circle(imgRight, selPointsRight[i], 3, cv::Scalar(255, 255, 255), 2);
			//circle(imgLeft, Point2f(np1.at<Vec2f>(0, i)), 3, cv::Scalar(255, 0, 0), 2);
			//circle(imgRight, Point2f(np2.at<Vec2f>(0, i)), 3, cv::Scalar(255, 0, 0), 2);
		//}
	}

	/* OpenCV féle */
//	Mat output4D;
//	triangulatePoints(magic.P1, magic.P2, projPointsLeft, projPointsRight, output4D);
//	Mat homogeneousPoints4d = output4D.t();
//	const int dimension = 4;
//	homogeneousPoints4d = homogeneousPoints4d.reshape(dimension);
//	Mat triangulatedPoints;
//	convertPointsFromHomogeneous(homogeneousPoints4d, triangulatedPoints);
//	writePLY3D(triangulatedPoints, projPointsLeft, imgLeftRM);

	/* HZ féle */
//	vector<Vec4d> homogeneousPoints;
//	for (size_t i = 0; i < projPointsLeft.size(); i++) {
//		Point2f u0 = projPointsLeft[i];
//		Point3d u(u0.x, u0.y, 0.0);

//		Point2f v0 = projPointsRight[i];
//		Point3d v(v0.x, v0.y, 0.0);

//		Mat res = IterativeLinearLSTriangulation(u, magic.P1, v, magic.P2);
//		Vec4d point(res);
//		homogeneousPoints.push_back(point);
//	}
//	writePLY3D(homogeneousPoints, projPointsLeft, imgLeft);

//	imwrite("left.jpg", imgLeftRM);
//	imwrite("right.jpg", imgRightRM);

//	cout << magic.P1;
//	cout.flush();

	//imshow("left", imgLeft);
	//imshow("right", imgRight);

	Mat points4D;
	cv::triangulatePoints(magic.P1, magic.P2, projPointsLeft, projPointsRight, points4D);
	//writePLY4D(points4D, projPointsLeft, imgLeftRM);

	while (true) {
		count++;

		try {
			// --------- OPENGL
			// Mat rvec, tvec;
			// magic.getCameraPose(rvec, tvec);
			//		renderer.updatePoints(magic.objectPoints, magic.imagePoints, imgLeftRM);
			// --------- END

			// --- MEGJELENÍTÉS
			canvas.put(imgLeft, 0);
			canvas.put(imgRight, 1);

			canvas.show("left -- right");

			//		renderer.render();

			int c = waitKey(10);
			if (c == ' ') {
				picIdx = ((picIdx + 1) % 8);
			} else if (c == 'k') {
				sc_type = ((sc_type + 1) % 3);
			} else if (c == 'l') {
				drawEpi = !drawEpi;
			} else if (c == 27) {
				break;
			} else if (c == 'e') {
				camera.rotX(10);
//			renderer.rotCameraX(5);
			} else if (c == 'q') {
				camera.rotX(-10);
//			renderer.rotCameraX(-5);
			} else if (c == 'd') {
				camera.rotY(10);
//			renderer.rotCameraY(5);
			} else if (c == 'a') {
				camera.rotY(-10);
//			renderer.rotCameraY(-5);
			} else if (c == 'c') {
				camera.rotZ(10);
			} else if (c == 'y') {
				camera.rotZ(-10);
			} else if (c == 'x') {
				camera.setRotation(0, 0, 0);
				camera.setTranslation(0.0, 0.0, 0.5);
			} else if (c == 'w') {
//			renderer.moveCamera(-1.0f);
				camera.addZ(-0.5);
			} else if (c == 's') {
//			renderer.moveCamera(1.0f);
				camera.addZ(0.5);
			} else if (c == 2424832) { // left
				camera.addX(0.5);
			} else if (c == 2555904) { // right
				camera.addX(-0.5);
			} else if (c == 2490368) { // up
				camera.addY(0.5);
			} else if (c == 2621440) { // down
				camera.addY(-0.5);
			}

		} catch (cv::Exception& ex) {

		}
	}

//	destroyAllWindows();

	return 0;
}
