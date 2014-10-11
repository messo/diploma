/*
 * PCDReader.h
 *
 *  Created on: 2014.04.02.
 *      Author: Balint
 */

#ifndef PCDREADER_H_
#define PCDREADER_H_

#include <opencv2/core/core.hpp>
#include <sstream>
#include <iostream>
#include <fstream>

class PCDReader {
public:
	PCDReader();
	virtual ~PCDReader();

	void read(const char * filename, cv::Mat& xyz, std::vector<int>& rgbas);
};

#endif /* PCDREADER_H_ */
