/*
 * PCDWriter.h
 *
 *  Created on: 2014.03.24.
 *      Author: Balint
 */

#ifndef PCDWRITER_H_
#define PCDWRITER_H_

#include <opencv2/core/core.hpp>
#include <iostream>
#include <fstream>

class PCDWriter {
public:
	PCDWriter();
	virtual ~PCDWriter();

	void write(const char * filename, const cv::Mat& xyz, const cv::Mat& img, double vx, double vy, double vz);

	void write(const char * filename, const cv::Mat& xyz, double vx, double vy, double vz);

	void write(const char * filename, const cv::Mat& xyz, const std::vector<int>& rgbas);
};

#endif /* PCDWRITER_H_ */
