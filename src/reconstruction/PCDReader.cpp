/*
 * PCDReader.cpp
 *
 *  Created on: 2014.04.02.
 *      Author: Balint
 */

#include "PCDReader.h"

PCDReader::PCDReader() {
	// TODO Auto-generated constructor stub

}

PCDReader::~PCDReader() {
	// TODO Auto-generated destructor stub
}

void PCDReader::read(const char * filename, cv::Mat& xyz, std::vector<int>& rgbas) {
	std::string line;
	int cols, rows, points;

	std::ifstream myfile;
	myfile.open(filename);
	getline(myfile, line); //myfile >> "VERSION .7";
	getline(myfile, line); //myfile >> "FIELDS x y z";
	getline(myfile, line); //myfile >> "SIZE 4 4 4";
	getline(myfile, line); //myfile >> "TYPE F F F";
	getline(myfile, line); //myfile >> "COUNT 1 1 1";
	myfile >> line >> cols;
	myfile >> line >> rows;
	myfile >> line; // enélkül fura, de szarni bele.
	getline(myfile, line); //myfile >> "VIEWPOINT";
	myfile >> line >> points;
	myfile >> line;
	getline(myfile, line); //myfile >> "DATA ascii";

	xyz.create(rows, cols, CV_32FC3);
	rgbas.resize(rows * cols);

	int ix = 0, iy = 0;
	while (getline(myfile, line)) {
		cv::Vec3f& point = xyz.at<cv::Vec3f>(iy, ix);
		if (line.compare("nan nan nan nan") == 0) {
			// NaN
			point[0] = 0.0f; // mindegy
			point[1] = 0.0f; // mindegy
			point[2] = -1.0f; // !!
			rgbas[iy * cols + ix] = 0;
		} else {
			std::stringstream ss;
			float x, y, z;
			int rgba;

			ss.str(line);
			ss >> x >> y >> z >> rgba;
			point[0] = x;
			point[1] = y;
			point[2] = z;
			rgbas[iy * cols + ix] = rgba;
		}
		if (++ix == 640) {
			ix = 0;
			++iy;
		}
	}

//	for (int y = 0; y < xyz.rows; y++) {
//		for (int x = 0; x < xyz.cols; x++) {
//			cv::Vec3f point = xyz.at<cv::Vec3f>(y, x);
//			if (point[2] == -1.0f) {
//				myfile << "nan nan nan" << std::endl;
//			} else {
//
//				myfile << point[0] << " " << point[1] << " " << point[2] << std::endl;
//			}
//		}
//	}

	myfile.close();
}
