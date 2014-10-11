/*
 * PCDWriter.cpp
 *
 *  Created on: 2014.03.24.
 *      Author: Balint
 */

#include "PCDWriter.h"

PCDWriter::PCDWriter() {
	// TODO Auto-generated constructor stub

}

PCDWriter::~PCDWriter() {
	// TODO Auto-generated destructor stub
}

void PCDWriter::write(const char * filename, const cv::Mat& xyz, double vx, double vy, double vz) {
	std::ofstream myfile;
	myfile.open(filename);
	myfile << "VERSION .7" << std::endl;
	myfile << "FIELDS x y z" << std::endl;
	myfile << "SIZE 4 4 4" << std::endl;
	myfile << "TYPE F F F" << std::endl;
	myfile << "COUNT 1 1 1" << std::endl;
	myfile << "WIDTH " << xyz.cols << std::endl;
	myfile << "HEIGHT " << xyz.rows << std::endl;
	myfile << "VIEWPOINT " << vx << " " << vy << " " << vz << " 1 0 0 0" << std::endl;
	myfile << "POINTS " << (xyz.cols * xyz.rows) << std::endl;
	myfile << "DATA ascii" << std::endl;

	for (int y = 0; y < xyz.rows; y++) {
		for (int x = 0; x < xyz.cols; x++) {
			cv::Vec3f point = xyz.at<cv::Vec3f>(y, x);
			if (point[2] == -1.0f) {
				myfile << "nan nan nan" << std::endl;
			} else {

				myfile << point[0] << " " << point[1] << " " << point[2] << std::endl;
			}
		}
	}

	myfile.close();
}

void PCDWriter::write(const char * filename, const cv::Mat& xyz, const std::vector<int>& rgbas) {
	std::ofstream myfile;
	myfile.open(filename);
	myfile << "VERSION .7" << std::endl;
	myfile << "FIELDS x y z rgba" << std::endl;
	myfile << "SIZE 4 4 4 4" << std::endl;
	myfile << "TYPE F F F U" << std::endl;
	myfile << "COUNT 1 1 1 1" << std::endl;
	myfile << "WIDTH " << xyz.cols << std::endl;
	myfile << "HEIGHT " << xyz.rows << std::endl;
	myfile << "VIEWPOINT 0 0 0 1 0 0 0" << std::endl;
	myfile << "POINTS " << (xyz.cols * xyz.rows) << std::endl;
	myfile << "DATA ascii" << std::endl;

	for (int y = 0; y < xyz.rows; y++) {
		for (int x = 0; x < xyz.cols; x++) {
			cv::Vec3f point = xyz.at<cv::Vec3f>(y, x);
			if (point[2] == -1.0f) {
				myfile << "0 0 0 0" << std::endl;
			} else {
				myfile << point[0] << " " << point[1] << " " << point[2] << " " << rgbas[y * xyz.cols + x] << std::endl;
			}
		}
	}

	myfile.close();
}

void PCDWriter::write(const char * filename, const cv::Mat& xyz, const cv::Mat& img, double vx, double vy, double vz) {
	std::ofstream myfile;
	myfile.open(filename);
	myfile << "VERSION .7" << std::endl;
	myfile << "FIELDS x y z rgba" << std::endl;
	myfile << "SIZE 4 4 4 4" << std::endl;
	myfile << "TYPE F F F U" << std::endl;
	myfile << "COUNT 1 1 1 1" << std::endl;
	myfile << "WIDTH " << xyz.cols << std::endl;
	myfile << "HEIGHT " << xyz.rows << std::endl;
	myfile << "VIEWPOINT " << vx << " " << vy << " " << vz << " 1 0 0 0" << std::endl;
	myfile << "POINTS " << (xyz.cols * xyz.rows) << std::endl;
	myfile << "DATA ascii" << std::endl;

	for (int y = 0; y < xyz.rows; y++) {
		for (int x = 0; x < xyz.cols; x++) {
			cv::Vec3f point = xyz.at<cv::Vec3f>(y, x);
			if (point[2] == -1.0f) {
				myfile << "nan nan nan nan" << std::endl;
			} else {
				cv::Vec3b color = img.at<cv::Vec3b>(y, x);
				int rgb = color[2] << 16 | color[1] << 8 | color[0];
				myfile << point[0] << " " << point[1] << " " << point[2] << " " << rgb << std::endl;
			}
		}
	}

	myfile.close();
}
