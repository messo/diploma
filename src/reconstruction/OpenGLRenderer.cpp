/*
 * OpenGLRenderer.cpp
 *
 *  Created on: 2014.03.20.
 *      Author: Balint
 */

#include "OpenGLRenderer.h"

OpenGLRenderer::OpenGLRenderer(int width, int height) :
	width(width), height(height)
{
	namedWindow("OpenGL", WINDOW_OPENGL);
	resizeWindow("OpenGL", width, height);

	cameraAngleX = CAMERA_ANGLEX;
	cameraAngleY = CAMERA_ANGLEY;
	cameraDistance = CAMERA_DISTANCE;
}

OpenGLRenderer::~OpenGLRenderer()
{
	destroyWindow("OpenGL");
}

void OpenGLRenderer::init()
{
	setOpenGlDrawCallback("OpenGL", openGlDrawCallbackFunc, this);
}

void OpenGLRenderer::render()
{
	updateWindow("OpenGL");
}

void OpenGLRenderer::openGlDrawCallbackFunc(void * userdata)
{
	OpenGLRenderer * ptr = static_cast<OpenGLRenderer*>(userdata);
	ptr->onDraw();
}

void OpenGLRenderer::updatePoints(const vector<Point3f>& points, const vector<Point2f>& imagePoints, const Mat& image)
{
	this->points = &points;
	this->imagePoints = &imagePoints;
	this->image = &image;
}

geom::Vector getVFromSC(float r, float theta, float phi)
{
	return geom::Vector(r * sinf(phi) * cosf(theta), r * cosf(phi), r * sinf(phi) * sinf(theta));
}

void OpenGLRenderer::setModelView(double * mat)
{
	this->mat = mat;
}

void OpenGLRenderer::setProjectionMatx(double * mat)
{
	this->projMat = mat;
}

void OpenGLRenderer::moveCamera(float delta)
{
	cameraDistance += delta;
}

void OpenGLRenderer::rotCameraX(float delta)
{
	cameraAngleX += delta;
}

void OpenGLRenderer::rotCameraY(float delta)
{
	cameraAngleY += delta;
}

void OpenGLRenderer::onDraw()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//glViewport(0, 0, width, height);

	glMatrixMode(GL_PROJECTION);

	//glLoadIdentity();
	//gluPerspective(45.0, (double) width / height, 0.0, 100.0);

	glLoadMatrixd(projMat);

	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixd(mat);

	//geom::Vector distance = getVFromSC(-cameraDistance,
	//		-cameraAngleY / 180.f * M_PI, -cameraAngleX / 180.f * M_PI);
	//geom::Vector from2 = from - distance;
	//gluLookAt(from2.x, from2.y, from2.z, 0.0f, 0.0f, 10.0f, 0, -1, 0);

	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

//	glBegin(GL_QUADS);        // Draw The Cube Using quads
//	glColor3f(0.0f, 1.0f, 0.0f);    // Color Blue
//	glVertex3f(1.0f, 1.0f, -1.0f);    // Top Right Of The Quad (Top)
//	glVertex3f(-1.0f, 1.0f, -1.0f);    // Top Left Of The Quad (Top)
//	glVertex3f(-1.0f, 1.0f, 1.0f);    // Bottom Left Of The Quad (Top)
//	glVertex3f(1.0f, 1.0f, 1.0f);    // Bottom Right Of The Quad (Top)
//	glColor3f(1.0f, 0.5f, 0.0f);    // Color Orange
//	glVertex3f(1.0f, -1.0f, 1.0f);    // Top Right Of The Quad (Bottom)
//	glVertex3f(-1.0f, -1.0f, 1.0f);    // Top Left Of The Quad (Bottom)
//	glVertex3f(-1.0f, -1.0f, -1.0f);    // Bottom Left Of The Quad (Bottom)
//	glVertex3f(1.0f, -1.0f, -1.0f);    // Bottom Right Of The Quad (Bottom)
//	glColor3f(1.0f, 0.0f, 0.0f);    // Color Red
//	glVertex3f(1.0f, 1.0f, 1.0f);    // Top Right Of The Quad (Front)
//	glVertex3f(-1.0f, 1.0f, 1.0f);    // Top Left Of The Quad (Front)
//	glVertex3f(-1.0f, -1.0f, 1.0f);    // Bottom Left Of The Quad (Front)
//	glVertex3f(1.0f, -1.0f, 1.0f);    // Bottom Right Of The Quad (Front)
//	glColor3f(1.0f, 1.0f, 0.0f);    // Color Yellow
//	glVertex3f(1.0f, -1.0f, -1.0f);    // Top Right Of The Quad (Back)
//	glVertex3f(-1.0f, -1.0f, -1.0f);    // Top Left Of The Quad (Back)
//	glVertex3f(-1.0f, 1.0f, -1.0f);    // Bottom Left Of The Quad (Back)
//	glVertex3f(1.0f, 1.0f, -1.0f);    // Bottom Right Of The Quad (Back)
//	glColor3f(0.0f, 0.0f, 1.0f);    // Color Blue
//	glVertex3f(-1.0f, 1.0f, 1.0f);    // Top Right Of The Quad (Left)
//	glVertex3f(-1.0f, 1.0f, -1.0f);    // Top Left Of The Quad (Left)
//	glVertex3f(-1.0f, -1.0f, -1.0f);    // Bottom Left Of The Quad (Left)
//	glVertex3f(-1.0f, -1.0f, 1.0f);    // Bottom Right Of The Quad (Left)
//	glColor3f(1.0f, 0.0f, 1.0f);    // Color Violet
//	glVertex3f(1.0f, 1.0f, -1.0f);    // Top Right Of The Quad (Right)
//	glVertex3f(1.0f, 1.0f, 1.0f);    // Top Left Of The Quad (Right)
//	glVertex3f(1.0f, -1.0f, 1.0f);    // Bottom Left Of The Quad (Right)
//	glVertex3f(1.0f, -1.0f, -1.0f);    // Bottom Right Of The Quad (Right)
//	glEnd();            // End Drawing The Cube

	glBegin(GL_POINTS);
	//for (int y = 0; y < points->rows; y += 1) {
//		for (int x = 0; x < points->cols; x += 1) {
	for (unsigned int i = 0; i < points->size(); i++) {
		const Point3f * point = &((*points)[i]);
		const Point2f * pos = &((*imagePoints)[i]);
		const Vec3b * col = &(image->at<Vec3b>((int)pos->y, (int)pos->x));
		glColor3ub(col->val[2], col->val[1], col->val[0]);
		glVertex3f(point->x, point->y, point->z);
	}
	//}
	//}
	glEnd();
	glFlush();

}
