/*
 * OpenGLRenderer.h
 *
 *  Created on: 2014.03.20.
 *      Author: Balint
 */

#ifndef OPENGLRENDERER_H_
#define OPENGLRENDERER_H_

#define _USE_MATH_DEFINES
#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include <GL/gl.h>
#include <GL/glu.h>
#include <iostream>
#include "Vector.h"

using namespace cv;
using namespace std;

const float CAMERA_DISTANCE = 0.0f;
const float CAMERA_ANGLEX = -90.0f;
const float CAMERA_ANGLEY = 90.0f;

class OpenGLRenderer {
	int width, height;
	const vector<Point3f> * points;
	const vector<Point2f> * imagePoints;
	const Mat * image;
	float cameraAngleX;
	float cameraAngleY;
	float cameraDistance;
	double * projMat;
	double * mat;
public:
	static void openGlDrawCallbackFunc(void* userdata);

	OpenGLRenderer(int width, int height);
	virtual ~OpenGLRenderer();

	void init();

	void render();

	void updatePoints(const vector<Point3f>& xyz, const vector<Point2f>& imagePoints, const Mat& image);

	void onDraw();

	void setProjectionMatx(double *);

	void setModelView(double *);

	void moveCamera(float delta);

	void rotCameraX(float delta);

	void rotCameraY(float delta);
};

#endif /* OPENGLRENDERER_H_ */
