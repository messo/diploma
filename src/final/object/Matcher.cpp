#include "Matcher.h"
#include "../Common.h"
#include "../PerformanceMonitor.h"

#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iomanip>

std::vector<std::pair<cv::Point2f, cv::Point2f>> Matcher::match(const std::vector<cv::Mat> &images,
                                                                const std::vector<cv::Mat> &masks) {
    double t0 = cv::getTickCount();

    detectKeypointsAndExtractDescriptors(images, masks);

    if (descriptors[0].empty() || descriptors[1].empty()) {
        return std::vector<std::pair<cv::Point2f, cv::Point2f>>();
    }

    std::vector<cv::DMatch> matches = matchDescriptors();

    std::vector<cv::DMatch> keptMatches;
    std::vector<std::pair<cv::Point2f, cv::Point2f>> result = buildMatches(matches, keptMatches);

    t0 = ((double) cv::getTickCount() - t0) / cv::getTickFrequency();
    std::cout << "[" << std::setw(20) << "Matcher" << "] " << "Matching done in " << t0 << "s" << std::endl;
    std::cout.flush();

    // DEBUG ------
    debug2(images, masks, keptMatches);

    // DEBUG ------

    return result;
}

void Matcher::debug2(const std::vector<cv::Mat> &images, const std::vector<cv::Mat> &masks, std::vector<cv::DMatch> &keptMatches) const {
    cv::Mat _frame1, _frame2;
    images[0].copyTo(_frame1, masks[0]);
    images[1].copyTo(_frame2, masks[1]);

//    cv::Rect left(120, 20, 400, 300);
//    cv::Rect right(220, 20, 400, 300);

    cv::Rect left(0, 0, 640, 480);
    cv::Rect right(0, 0, 640, 480);


    std::vector<cv::Point2f> points1, points2;
    for (int i = 0; i < this->keypoints[0].size(); i++) {
        points1.push_back(this->keypoints[0][i].pt - cv::Point2f(left.tl()));
    }
    for (int i = 0; i < this->keypoints[1].size(); i++) {
        points2.push_back(this->keypoints[1][i].pt - cv::Point2f(right.tl()));
    }


//    sort(keptMatches.begin(), keptMatches.end());
//    keptMatches.erase(keptMatches.begin(), keptMatches.end() - 20);

    cv::Mat img_matches = mergeImages(_frame1(left), _frame2(right));
    cv::RNG rng;

    cv::line(img_matches, cv::Point(left.width - 1, 0), cv::Point(left.width - 1, left.height), cv::Scalar(255, 255, 255));
    cv::line(img_matches, cv::Point(left.width, 0), cv::Point(left.width, left.height), cv::Scalar(255, 255, 255));

    for (int i = 0; i < keptMatches.size(); i++) {
        cv::Point p1(points1[keptMatches[i].queryIdx]);
        cv::Point p2(points2[keptMatches[i].trainIdx]);

        int icolor = (unsigned) rng;
        cv::Scalar color(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);

        cv::line(img_matches, p1, p2 + cv::Point(left.width, 0), color, 1, cv::LINE_AA);
        cv::circle(img_matches, p1, 3, color, 1, cv::LINE_AA);
        cv::circle(img_matches, p2 + cv::Point(left.width, 0), 3, color, 1, cv::LINE_AA);
    }



//    drawMatches(_frame1(left), keyPoints1, _frame2(right), keyPoints2, keptMatches, img_matches,
//                ::cv::Scalar_<double>::all(-1), ::cv::Scalar_<double>::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("Matches", img_matches);
    imwrite("/media/balint/Data/Linux/multi_obj_matches_bad.png", img_matches);
}

void Matcher::detectKeypointsAndExtractDescriptors(const std::vector<cv::Mat> &images,
                                                   const std::vector<cv::Mat> &masks) {
    double t0 = cv::getTickCount();

    PerformanceMonitor::get()->extractingStarted();

    cv::OrbFeatureDetector detector;
//    cv::FastFeatureDetector detector(100);
//    cv::xfeatures2d::SURF detector(600);
//    cv::BRISK detector;

//    cv::xfeatures2d::FREAK extractor;
    cv::OrbDescriptorExtractor extractor;

//    cv::xfeatures2d::SURF extractor;

    keypoints.resize(images.size());
    descriptors.resize(images.size());

    for (int i = 0; i < images.size(); i++) {
        detector.detect(images[i], keypoints[i], masks[i]);
        extractor.compute(images[i], keypoints[i], descriptors[i]);
    }

    t0 = ((double) cv::getTickCount() - t0) / cv::getTickFrequency();
    std::cout << "[" << std::setw(20) << "Matcher" << "] " << "Detection and extraction done in " << t0 << "s" << std::endl;
    std::cout.flush();

    PerformanceMonitor::get()->extractingFinished();
}

std::vector<cv::DMatch> Matcher::matchDescriptors() {
//    cv::FlannBasedMatcher matcher; //(cv::Ptr<cv::flann::IndexParams>(new cv::flann::LshIndexParams(20, 10, 2)));
//    std::vector<cv::DMatch> matches;
//    matcher.match(descriptors[0], descriptors[1], matches);

    cv::BFMatcher matcher(cv::NORM_HAMMING);

    std::vector<cv::DMatch> matches;
    matcher.match(descriptors[0], descriptors[1], matches);

//    std::vector<std::vector<cv::DMatch>> nn_matches;
//    matcher.knnMatch(descriptors[0], descriptors[1], nn_matches, 2);
//
//    std::vector<cv::DMatch> matches;
//    for (auto it = nn_matches.begin(); it != nn_matches.end(); ++it) {
//            matches.push_back((*it)[0]);
//    }

    return matches;

//    double max_dist = 0;
//    double min_dist = 100;
//
//    //-- Quick calculation of max and min distances between keypoints
//    for (int i = 0; i < descriptors[0].rows; i++) {
//        double dist = matches[i].distance;
//        if (dist < min_dist) min_dist = dist;
//        if (dist > max_dist) max_dist = dist;
//    }
//
//    printf("-- Max dist : %f \n", max_dist);
//    printf("-- Min dist : %f \n", min_dist);
//
//    std::vector<cv::DMatch> good_matches;
//    std::cout << "Matches before distnace selection: " << matches.size() << std::endl;
//    for (int i = 0; i < descriptors[0].rows; i++) {
//        if (matches[i].distance <= cv::max(2 * min_dist, 0.02)) {
//            good_matches.push_back(matches[i]);
//        }
//    }
//
//    return good_matches;
}

std::vector<std::pair<cv::Point2f, cv::Point2f>> Matcher::buildMatches(const std::vector<cv::DMatch> &matches,
                                                                       std::vector<cv::DMatch> &keptMatches) {

    std::vector<cv::Point2f> points1, points2;
    for (auto it = matches.begin(); it != matches.end(); ++it) {
        // Get the position of left keypoints
        float x = keypoints[0][it->queryIdx].pt.x;
        float y = keypoints[0][it->queryIdx].pt.y;
        points1.push_back(cv::Point2f(x, y));
        // Get the position of right keypoints
        x = keypoints[1][it->trainIdx].pt.x;
        y = keypoints[1][it->trainIdx].pt.y;
        points2.push_back(cv::Point2f(x, y));
    }

    std::vector<std::pair<cv::Point2f, cv::Point2f>> matchingPoints;

    if (points1.empty() || points2.empty()) {
        return matchingPoints;
    }

    std::vector<cv::Point2f> u_points1, u_points2;
    cv::undistortPoints(points1, u_points1, camera1->cameraMatrix, camera1->distCoeffs, cv::Mat(),
                        camera1->cameraMatrix);
    cv::undistortPoints(points2, u_points2, camera2->cameraMatrix, camera2->distCoeffs, cv::Mat(),
                        camera2->cameraMatrix);

    for (int i = 0; i < u_points1.size(); i++) {
        cv::Mat res =
                cv::Mat(cv::Vec3d(u_points2[i].x, u_points2[i].y, 1).t()) * F *
                cv::Mat(cv::Vec3d(u_points1[i].x, u_points1[i].y, 1));

        if (fabs(res.at<double>(0)) < THRESHOLD) {
            keptMatches.push_back(matches[i]);
            matchingPoints.push_back(std::make_pair(points1[i], points2[i]));
        }
    }

    std::cout << "[" << std::setw(20) << "Matcher" << "] " << keptMatches.size() << "/" << matches.size() << " has been kept by F." << std::endl;

    return matchingPoints;
}
