#pragma once

#include "StereoCamera.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cmath>
#include <GL/gl.h>
#include <GL/glu.h>
#include <iostream>

const float CAMERA_DISTANCE = 0.0f;
const float CAMERA_ANGLEX = -90.0f;
const float CAMERA_ANGLEY = 90.0f;

class OpenGLRenderer {
    int width, height;

    cv::Ptr<StereoCamera> stereoCamera;
    cv::Ptr<cv::Mat> projectionMatrix;
    cv::Ptr<cv::Mat> modelViewMatrix;

    std::vector<cv::Point3f> *points;
    std::vector<cv::Point2f> *imagePoints;
    cv::Mat *image;

    double centerX = 0, centerY = 0;
    double cameraAngleX;
    double cameraAngleY;
    double cameraDistance;
public:
    static void openGlDrawCallbackFunc(void *userdata);

    OpenGLRenderer(cv::Ptr<StereoCamera> stereoCamera);

    virtual ~OpenGLRenderer();

    void init();

    void render();

    void updatePoints(std::vector<cv::Point3f> *xyz, std::vector<cv::Point2f> *imagePoints, cv::Mat *image);

    void onDraw();

    void setProjection(const cv::Mat &cam);

    void setModelView(const cv::Mat &rvec, const cv::Mat &tvec);

    void moveCamera(double delta);

    void rotCameraX(double delta);

    void rotCameraY(double delta);

    void moveCenterY(double d);

    void moveCenterX(double d);

    void resetCamera();
};
