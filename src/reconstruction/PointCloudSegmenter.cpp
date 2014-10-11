/*
 * PointCloudSegmenter.cpp
 *
 *  Created on: 2014.03.10.
 *      Author: Balint
 */

#include "PointCloudSegmenter.h"

PointCloudSegmenter::PointCloudSegmenter() {
	// TODO Auto-generated constructor stub

}

void checkedAdd(Point& p, const Mat& disparity, int x, int y,
		std::queue<Point>& queue, Mat& seen) {
	if (x < 0 || x >= seen.cols || y < 0 || y >= seen.rows)
		return;

	if (seen.at<uchar>(y, x) == 0) { // ha még nem láttuk
		seen.at<uchar>(y, x) = 255;

		// most jön az ellenõrzés, hogy valóban jó-e, akkor jó, ha a távolsága kicsi
		uchar prev = disparity.at<uchar>(p);
		uchar now = disparity.at<uchar>(y, x);

		if (abs(now - prev) < Z_DELTA) {
			queue.push(Point(x, y));
		}
	}
}

Mat PointCloudSegmenter::segment(const Mat& disparity, const Point& p, vector<Point>& output) {
	vector<Point> choosen;

	// Point p(140, 390); // kiinduló pont a számítógéphez tartozó pontfelhõhöz.
	// Point p(300, 180); // emberke feje

	Mat magic(disparity);

	Mat seen(disparity.rows, disparity.cols, CV_8UC1);
	for (int i = 0; i < seen.rows; i++) {
		for (int j = 0; j < seen.cols; j++) {
			seen.at<uchar>(i, j) = 0;
		}
	}

	// mehet az elárasztás
	std::queue<Point> queue;
	queue.push(p); // kiindulópont
	seen.at<uchar>(p) = 255;

	while (!queue.empty()) {
		Point p = queue.front();
		queue.pop();

		// szomszédok
		for (int d = -1; d <= 1; d += 2) {
			checkedAdd(p, disparity, p.x, p.y + d, queue, seen);
			checkedAdd(p, disparity, p.x + d, p.y, queue, seen);
		}
	}

	return seen;
}

PointCloudSegmenter::~PointCloudSegmenter() {
	// TODO Auto-generated destructor stub
}

