/*
 * merge_clouds.cpp
 *
 *  Created on: 2014.04.02.
 *      Author: Balint
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "PCDReader.h"
#include "PCDWriter.h"

using namespace cv;
using namespace cv::xfeatures2d;

double absvec(const Vec3d& vec)
{
	return sqrt(vec.dot(vec));
}

int main()
{
	PCDReader reader;
	PCDWriter writer;

	Mat balImg(imread("bal.jpg")), jobbImg(imread("jobb.jpg"));

	double minHessian = 300.0;
	SIFT sift(0, 3, 0.04, 20, 0.5);
	SURF surf(minHessian);

	std::vector<KeyPoint> keypoints_left, keypoints_right;
	Mat descriptors_left, descriptors_right;

	Mat mask_left = Mat::zeros(balImg.size(), CV_8U);
	Mat roi_left(mask_left, Rect(450, 100, 600 - 450, 420 - 100));
	roi_left = Scalar(255, 255, 255);
	surf(balImg, mask_left, keypoints_left, descriptors_left, false);

	Mat mask_right = Mat::zeros(jobbImg.size(), CV_8U);
	Mat roi_right(mask_right, Rect(100, 150, 280 - 100, 470 - 150));
	roi_right = Scalar(255, 255, 255);
	surf(jobbImg, mask_right, keypoints_right, descriptors_right, false);

	FlannBasedMatcher matcher;
	std::vector<DMatch> matches;
	matcher.match(descriptors_left, descriptors_right, matches);

	//Mat output;
	//drawMatches(balImg, keypoints_left, jobbImg, keypoints_right, matches, output);
	//imshow("output", output);
	//waitKey();

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
	for (std::vector<DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it) {
		// Get the indexes of the selected matchedkeypoints
		if (it->distance <= max(1.5f * min_dist, 0.02)) {
			pointIndexesLeft.push_back(it->queryIdx);
			pointIndexesRight.push_back(it->trainIdx);
		}
	}

	std::vector<Point2f> selPointsLeft, selPointsRight;
	KeyPoint::convert(keypoints_left, selPointsLeft, pointIndexesLeft);
	KeyPoint::convert(keypoints_right, selPointsRight, pointIndexesRight);

	std::vector<Point2f> projPointsLeft, projPointsRight;
	RNG& rng = theRNG();
	for (size_t i = 0; i < selPointsLeft.size(); i++) {
		Scalar color = Scalar(rng(256), rng(256), rng(256));

		projPointsLeft.push_back(selPointsLeft[i]);
		projPointsRight.push_back(selPointsRight[i]);
		circle(balImg, selPointsLeft[i], 3, color, 2);
		circle(jobbImg, selPointsRight[i], 3, color, 2);
	}

	imshow("left", balImg);
	imshow("jobb", jobbImg);
	waitKey();

	Mat bal, jobb;
	std::vector<int> balRGBA, jobbRGBA;
	reader.read("bal.pcd", bal, balRGBA);
	reader.read("jobb.pcd", jobb, jobbRGBA);

	std::vector<Vec3d> left3ds, right3ds;
	std::cout << "Matches:" << std::endl;
	for (size_t i = 0; i < projPointsLeft.size(); i++) {
		Point2f left(projPointsLeft[i]);
		Point2f right(projPointsRight[i]);

		Vec3d left3d(bal.at<Vec3f>(left));
		Vec3d right3d(jobb.at<Vec3f>(right));

		if (left3d[2] != -1.0f && right3d[2] != -1.0f && i != 2) {
			// valid 3D-s pont, mehet!
			left3ds.push_back(left3d);
			right3ds.push_back(right3d);
			std::cout << left3d << " " << right3d << std::endl;
		}
	}

	double ratioSum = 0.0;
	int count = 0;

	for (size_t i = 0; i < left3ds.size(); i++) {
		for (size_t j = i + 1; j < left3ds.size(); j++) {
			if (j == i)
				continue;

			double lleft = absvec(left3ds[i] - left3ds[j]);
			double lright = absvec(right3ds[i] - right3ds[j]);
			double ratio = lleft / lright;
			std::cout << i << "-" << j << ": " << lleft << " " << lright << " ratio: " << ratio << std::endl;
			ratioSum += ratio;
			++count;
		}
	}

	double ratio = ratioSum / count;
	ratio = 1.0;

	std::vector<Vec3d> rightZoomed3ds;
	for (size_t i = 0; i < right3ds.size(); i++) {
		rightZoomed3ds.push_back(right3ds[i] * ratio);
	}

	Mat affineMat, inliers;
	estimateAffine3D(left3ds, rightZoomed3ds, affineMat, inliers);
	std::cout << ratio << " IN: " << inliers << std::endl;
	std::cout << affineMat << std::endl;

	// affin trafó:

//	affineMat.at<double>(0, 0) = 1.0;
//	affineMat.at<double>(0, 1) = 0.0;
//	affineMat.at<double>(0, 2) = 0.0;
//	affineMat.at<double>(0, 3) = 0.0;
//
//	affineMat.at<double>(1, 0) = 0.0;
//	affineMat.at<double>(1, 1) = 1.0;
//	affineMat.at<double>(1, 2) = 0.0;
//	affineMat.at<double>(1, 3) = 0.0;
//
//	affineMat.at<double>(2, 0) = 0.0;
//	affineMat.at<double>(2, 1) = 0.0;
//	affineMat.at<double>(2, 2) = 1.0;
//	affineMat.at<double>(2, 3) = 0.0;

	// ICP:

//	affineMat.at<double>(0, 0) = 0.524963;
//	affineMat.at<double>(0, 1) = -0.100537;
//	affineMat.at<double>(0, 2) = 0.845168;
//	affineMat.at<double>(0, 3) = -13.6256;
//
//	affineMat.at<double>(1, 0) = 0.715051;
//	affineMat.at<double>(1, 1) = 0.590697;
//	affineMat.at<double>(1, 2) = -0.373874;
//	affineMat.at<double>(1, 3) = 4.11708;
//
//	affineMat.at<double>(2, 0) = -0.461649;
//	affineMat.at<double>(2, 1) = 0.800607;
//	affineMat.at<double>(2, 2) = 0.381984;
//	affineMat.at<double>(2, 3) = 8.23136;

	Mat balAffin(bal.rows, bal.cols, CV_32FC3);
	Mat jobbAffin(jobb.rows, jobb.cols, CV_32FC3);
	for (int i = 0; i < bal.rows; i++) {
		for (int j = 0; j < bal.cols; j++) {
			Vec3f origF_L = bal.at<Vec3f>(i, j);
			Mat orig_L(Vec4d(origF_L[0], origF_L[1], origF_L[2], 1.0));

			if (origF_L[2] != -1.0f) {
				Vec3d tmp(Mat(affineMat * orig_L));
				balAffin.at<Vec3f>(i, j) = Vec3f((float) tmp[0], (float) tmp[1], (float) tmp[2]);
			} else {
				balAffin.at<Vec3f>(i, j) = Vec3f(0.0f, 0.0f, -1.0f);
			}

			Vec3f origF_R = jobb.at<Vec3f>(i, j);
			if (origF_R[2] != -1.0f) {
				jobbAffin.at<Vec3f>(i, j) = origF_R * ratio;
			} else {
				jobbAffin.at<Vec3f>(i, j) = Vec3f(0.0f, 0.0f, -1.0f);
			}
		}
	}

	writer.write("balAffin.pcd", balAffin, balRGBA);
	writer.write("jobbAffin.pcd", jobbAffin, jobbRGBA);

	return 0;
}
