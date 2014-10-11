/*
 * PointCloudSegmenter.h
 *
 *  Created on: 2014.03.10.
 *      Author: Balint
 */

#ifndef POINTCLOUDSEGMENTER_H_
#define POINTCLOUDSEGMENTER_H_

#include <queue>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

const int Z_DELTA = 5;

class PointCloudSegmenter {
public:
	PointCloudSegmenter();
	virtual ~PointCloudSegmenter();

	Mat segment(const Mat& disparity, const Point& p, vector<Point>& output);
};

#endif /* POINTCLOUDSEGMENTER_H_ */
