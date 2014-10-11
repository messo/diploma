
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching/warpers.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

//#include <opencv2/core/opengl_interop.hpp>

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

void writePLY(const Mat& xyz, const Mat& img) {
	FILE* fp = fopen("cloud.ply", "wt");
	fprintf(fp, "%s", "ply\n");
	fprintf(fp, "%s", "format ascii 1.0\n");

	int count = 0;
	for (int y = 0; y < xyz.rows; y += 4) {
		for (int x = 0; x < xyz.cols; x += 4) {
			Vec3f point = xyz.at<Vec3f>(y, x);
			if (point[2] == -1.0f)
				continue;
			count++;
		}
	}

	fprintf(fp, "element vertex %d\n", count);
	fprintf(fp, "%s", "property float x\n");
	fprintf(fp, "%s", "property float y\n");
	fprintf(fp, "%s", "property float z\n");
	fprintf(fp, "%s", "property uchar blue\n");
	fprintf(fp, "%s", "property uchar green\n");
	fprintf(fp, "%s", "property uchar red\n");
	fprintf(fp, "%s", "end_header\n");

	for (int y = 0; y < xyz.rows; y += 4) {
		for (int x = 0; x < xyz.cols; x += 4) {
			Vec3f point = xyz.at<Vec3f>(y, x);
			if (point[2] == -1.0f)
				continue;
			Vec3b s = img.at<Vec3b>(y, x);
			fprintf(fp, "%f %f %f %d %d %d\n", point[0], point[1], point[2], s[0], s[1], s[2]);
		}
	}

	std::cout << xyz.rows << " -- " << xyz.cols << "\n";
	std::cout.flush();

	fclose(fp);
}

void doMagic(Magic& magic, const Mat& imgLeftRM, const Camera& camera, const Mat& xyz, Mat& test, Mat& warped) {
	// ---------------------------
	magic.drawTest(test, imgLeftRM, camera);

	// ---------------------------
	// TRAFÓ!
	// ---------------------------
	// Random pont sorsolás
	vector<Point3f> objectPoints;
	vector<Point2f> imagePoints;
	magic.getPointsForTransformation(objectPoints, imagePoints, camera);
	std::cout << imagePoints.size() << "\n";
	std::cout.flush();

	// 1. pontok vetítése
	vector<Point2f> output;
	magic.projectPoints(objectPoints, camera, output);

	// 2. trafó
	Mat homographyMat = findHomography(imagePoints, output, RANSAC);
	warpPerspective(imgLeftRM, warped, homographyMat, imgLeftRM.size());
}

void doPointCloud(Magic& magic, Point p, const Mat& imgDisparity16S, Mat& xyz) {
	vector<Point> pc;

	PointCloudSegmenter pcs;
	Mat magicMat = pcs.segment(magic.normalizeDisparity(imgDisparity16S), p, pc);

	// imshow("magic", magicMat);
	Mat only(imgDisparity16S.rows, imgDisparity16S.cols, imgDisparity16S.type());
	for (int y = 0; y < only.rows; y++) {
		for (int x = 0; x < only.cols; x++) {
			if (magicMat.at<uchar>(y, x) != 255) {
				only.at<short>(y, x) = -30000;
			} else {
				only.at<short>(y, x) = imgDisparity16S.at<short>(y, x);
			}
		}
	}

	magic.reprojectTo3D(only, xyz);
}

int main(int argc, char** argv) {
	Magic magic;
	Canvas canvas(Size(640, 480), 2);

	Camera camera;
	camera.setTranslation(0.0, 0.0, 0.5);

	Mat imgLeftRM = magic.readAndRemap("left01.jpg", LEFT);
	Mat imgRightRM = magic.readAndRemap("right01.jpg", RIGHT);

	imwrite("left01_MAPPED.jpg", imgLeftRM);

	// ---------------------------
	// 3D projekció
	// ---------------------------
	Mat imgDisparity16S;
	magic.getDisparityMatrix(imgLeftRM, imgRightRM, SGBM, imgDisparity16S);
	Mat xyz;

	magic.reprojectTo3D(imgDisparity16S, xyz);
	Mat disp = magic.normalizeDisparity(imgDisparity16S);
	imshow("disp", disp);
	imwrite("disparity.jpg", disp(magic.dispRoi));

	// waitKey();

	// ---------------------------
	// Túl távoli képek kiszûrése
	// ---------------------------
	magic.filter3DCoordinates(xyz, magic.dispRoi);

	Mat rvec, tvec, rot;
	magic.getCameraPose(rvec, tvec);
	Rodrigues(rvec, rot);

	Mat output(Mat::zeros(imgLeftRM.rows, imgLeftRM.cols, CV_8UC3));
	cout << output.type() << " " << imgLeftRM.type();
	magic.reprojectPoints(rvec, tvec, imgLeftRM, output);
	imwrite("reproject.jpg", output);

	// --- MEGJELENÍTÉS
	canvas.put(imgLeftRM, 0);
	canvas.put(imgRightRM, 1);
	// canvas.put(test, 2);

	canvas.show("left -- point-cloud -- warped");

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
	cvToGl.at<double>(3, 3) = 1.0f;
	RT = cvToGl * RT;

	Mat glViewMatrix;
	transpose(RT, glViewMatrix);

	OpenGLRenderer renderer(640, 480);
	renderer.init();
	renderer.setModelView(&glViewMatrix.at<double>(0, 0));
	Mat projMat = magic.getGlProjectionMatrix();
	renderer.setProjectionMatx(&projMat.at<double>(0, 0));
	renderer.updatePoints(magic.objectPoints, magic.imagePoints, imgLeftRM);

	while (true) {
		renderer.render();

		char ch = waitKey(500);
		if (ch == 27) {
			break;
		} else if (ch == 'f') {
			PCDWriter w;
			Vec3d v(tvec);
			std::cout << v << std::endl;
			std::cout.flush();
			w.write("magic.pcd", xyz, imgLeftRM, v[0], v[1], v[2]);
			writePLY(xyz, imgLeftRM);
		}
	}

	return 0;
}
