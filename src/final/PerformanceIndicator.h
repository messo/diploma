#pragma once

#include <iostream>
#include <iomanip>
#include "opencv2/core.hpp"


class PerformanceIndicator {

    std::vector<double> items;

public:

    void addNew(double duration) {
        items.push_back(duration);
    }

    void print(const std::string &label) {
        cv::Scalar mean, stdDev;
        cv::meanStdDev(items, mean, stdDev);
        double minVal, maxVal;
        cv::minMaxLoc(items, &minVal, &maxVal);

        //std::cout << "%" << std::setw(30) << label << ": avg: " << mean[0] << ", min: " << minVal << ", max: " << maxVal << ", stddev: " << stdDev[0] << std::endl;
        std::cout << label << " & " << std::setprecision(3) << minVal << " & " << maxVal << " & " << mean[0] << " & " << stdDev[0] << " \\\\" << std::endl;
    }

};



