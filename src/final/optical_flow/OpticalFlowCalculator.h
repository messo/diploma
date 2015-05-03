#pragma once

#include <opencv2/core.hpp>
#include "../camera/Camera.hpp"
#include "../object/Object.h"

class OpticalFlowCalculator {

protected:

    double EPSILON_TEXTURE = 6.0;

    cv::Mat frames[2];
    cv::Mat masks[2];
    cv::Mat texturedRegions[2];

    cv::Mat calcTexturedRegion(const cv::Mat frame, const cv::Mat mask) const;

    virtual std::vector<cv::Mat> calcOpticalFlows() const;

    virtual std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> collectMatchingPoints(const std::vector<cv::Mat> &flows) const;


    void shiftFrame(int i, cv::Point shift);


    /** VISUALIZATIONS */

    void visualizeOpticalFlow(const cv::Mat &img1, const cv::Mat &mask1,
                              const cv::Mat &img2, const cv::Mat &mask2,
                              const cv::Mat &flow, const std::string &name) const;

    void visualizeMatches(const std::vector<cv::Point2f> &points1, const std::vector<cv::Point2f> &points2) const;

    void visualizeMatches(const cv::Mat &img1, const std::vector<cv::Point2f> &points1,
                          const cv::Mat &img2, const std::vector<cv::Point2f> &points2) const;

    void visualizeMatchesROI(cv::Mat const &img1, std::vector<cv::Point2f> const &points1, cv::Mat const &img2,
                             std::vector<cv::Point2f> const &points2);

public:

    virtual std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> calcDenseMatches(std::vector<cv::Mat> &frames, const Object &object);

};
