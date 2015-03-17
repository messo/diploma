#include "OpenGLRenderer.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "Vector.hpp"
#include <GL/glut.h>

using namespace std;
using namespace cv;

OpenGLRenderer::OpenGLRenderer(Ptr<StereoCamera> stereoCamera) : stereoCamera(stereoCamera), width(640), height(480) {
    namedWindow("OpenGL", WINDOW_OPENGL);
    resizeWindow("OpenGL", width, height);
    glEnable(GL_DEPTH_TEST);

    resetCamera();
}

OpenGLRenderer::~OpenGLRenderer() {
    destroyWindow("OpenGL");
}

void OpenGLRenderer::init() {
    setOpenGlDrawCallback("OpenGL", openGlDrawCallbackFunc, this);
}

void OpenGLRenderer::render() {
    updateWindow("OpenGL");
}

void OpenGLRenderer::openGlDrawCallbackFunc(void *userdata) {
    OpenGLRenderer *ptr = static_cast<OpenGLRenderer *>(userdata);
    ptr->onDraw();
}

void OpenGLRenderer::updatePoints(vector<Point3f> *points, vector<Point2f> *imagePoints, Mat *image) {
    this->points = points;
    this->imagePoints = imagePoints;
    this->image = image;

    if (modelViewMatrix.empty()) {
        Mat rvec, tvec;
        stereoCamera->getCameraPose(rvec, tvec);
        setModelView(rvec, tvec);
    }
}

void OpenGLRenderer::setModelView(const Mat &rvec, const Mat &tvec) {
    Mat rot;
    Rodrigues(rvec, rot);

    Mat RT = Mat::zeros(4, 4, rot.type());
    for (unsigned int row = 0; row < 3; ++row) {
        for (unsigned int col = 0; col < 3; ++col) {
            RT.at<double>(row, col) = rot.at<double>(row, col);
        }
        RT.at<double>(row, 3) = tvec.at<double>(row, 0);
    }
    RT.at<double>(3, 3) = 1.0f;

    cv::Mat cvToGl = cv::Mat::zeros(4, 4, CV_64F);
    cvToGl.at<double>(0, 0) = 1.0f;
    // Invert the y axis
    cvToGl.at<double>(1, 1) = -1.0f;
    // invert the z axis
    cvToGl.at<double>(2, 2) = -1.0f;
    // leave the translation untoched
    cvToGl.at<double>(3, 3) = 1.0f;
    RT = cvToGl * RT;

    Mat glViewMatrix;
    transpose(RT, glViewMatrix);

    modelViewMatrix = Ptr<Mat>(new Mat());
    glViewMatrix.copyTo(*modelViewMatrix.get());
}

void OpenGLRenderer::setProjection(const Mat &cam) {
    projectionMatrix = Ptr<Mat>(new Mat(4, 4, CV_64FC1));
    float far = 100.0f, near = 0.5f;
    projectionMatrix->at<double>(0, 0) = 2 * cam.at<double>(0, 0) / 640.;
    projectionMatrix->at<double>(0, 2) = -1 + (2 * cam.at<double>(0, 2) / 640.);
    projectionMatrix->at<double>(1, 1) = 2 * cam.at<double>(1, 1) / 480.;
    projectionMatrix->at<double>(1, 2) = -1 + (2 * cam.at<double>(1, 2) / 480.);
    projectionMatrix->at<double>(2, 2) = -(far + near) / (far - near);
    projectionMatrix->at<double>(2, 3) = -2 * far * near / (far - near);
    projectionMatrix->at<double>(3, 2) = -1;
}

void OpenGLRenderer::moveCamera(double delta) {
    cameraDistance += delta;
}

void OpenGLRenderer::rotCameraX(double delta) {
    cameraAngleX += delta;
}

void OpenGLRenderer::rotCameraY(double delta) {
    cameraAngleY += delta;
}

geom::Vector getVFromSC(double r, double theta, double phi) {
    return geom::Vector(r * sin(phi) * cos(theta), r * cos(phi), r * sin(phi) * sin(theta));
}

void OpenGLRenderer::onDraw() {
    if (imagePoints == NULL) {
        return;
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //glViewport(0, 0, width, height);

    glMatrixMode(GL_PROJECTION);
    glLoadMatrixd(&(projectionMatrix->at<double>(0, 0)));

    glMatrixMode(GL_MODELVIEW);
    //glLoadMatrixd(&(modelViewMatrix->at<double>(0, 0)));

    geom::Vector from(0, 0, 0);
    geom::Vector distance = getVFromSC(-cameraDistance,
            -cameraAngleY / 180.f * M_PI, -cameraAngleX / 180.f * M_PI);
    geom::Vector from2 = from - distance;
    glLoadIdentity();
    gluLookAt(from2.x, from2.y, from2.z, centerX, centerY, 0.0f, 0, -1, 0);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    static float minX = (*points)[0].x, maxX = (*points)[0].x;
    static float minY = (*points)[0].y, maxY = (*points)[0].y;
    static float minZ = (*points)[0].z, maxZ = (*points)[0].z;

    glBegin(GL_POINTS);
    for (unsigned int i = 0; i < points->size(); i++) {
        const Point3f *point = &((*points)[i]);
        const Point2f *pos = &((*imagePoints)[i]);
        const Vec3b *col = &(image->at<Vec3b>((int) pos->y, (int) pos->x));
        glColor3ub(col->val[2], col->val[1], col->val[0]);
        glVertex3f(point->x, point->y, point->z);

        minX = min(point->x, minX);
        maxX = max(point->x, maxX);
        minY = min(point->y, minY);
        maxY = max(point->y, maxY);
        minZ = min(point->z, minZ);
        maxZ = max(point->z, maxZ);
    }
    glEnd();

    glPushMatrix();
    glTranslatef((maxX + minX) / 2, (maxY + minY) / 2, (maxZ + minZ) / 2);
    glColor3ub(255, 255, 255);
    glutWireCube(max(max((maxX - minX), (maxY - minY)), (maxZ - minZ)));
    glPopMatrix();

    glFlush();
}

void OpenGLRenderer::moveCenterY(double d) {
    centerY += d;
}

void OpenGLRenderer::moveCenterX(double d) {
    centerX += d;
}

void OpenGLRenderer::resetCamera() {
    centerX = 0.0;
    centerY = 0.0;
    cameraAngleX = CAMERA_ANGLEX;
    cameraAngleY = CAMERA_ANGLEY;
    cameraDistance = CAMERA_DISTANCE;
}
