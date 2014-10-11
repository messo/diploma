/*
 * Canvas.cpp
 *
 *  Created on: 2014.03.09.
 *      Author: Balint
 */

#include "Canvas.h"

Canvas::Canvas(Size imageSize, int imgs) : imageSize(imageSize), imgs(imgs) {
	double sf = 640. / MAX(imageSize.width, imageSize.height);
	int w = cvRound(imageSize.width * sf);
	int h = cvRound(imageSize.height * sf);
	canvas.create(h, w * imgs, CV_8UC3);
}

void Canvas::put(const Mat& img, int index) {
	int w = canvas.cols / imgs;
	int h = canvas.rows;
	Mat canvasPart = canvas(Rect(w * index, 0, w, h));
	resize(img, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
}

Canvas::~Canvas() {
	// TODO Auto-generated destructor stub
}

