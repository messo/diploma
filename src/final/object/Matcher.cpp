#include "Matcher.h"

#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

std::vector<std::pair<cv::Point2f, cv::Point2f>> Matcher::match(const std::vector<cv::Mat> &images,
                                                                const std::vector<cv::Mat> &masks) {
    double t0 = cv::getTickCount();

    detectKeypointsAndExtractDescriptors(images, masks);

    std::vector<cv::DMatch> matches = matchDescriptors();

    std::vector<cv::DMatch> keptMatches;
    std::vector<std::pair<cv::Point2f, cv::Point2f>> result = buildMatches(matches, keptMatches);

    t0 = ((double) cv::getTickCount() - t0) / cv::getTickFrequency();
    std::cout << "- Matching done in " << t0 << "s" << std::endl;
    std::cout.flush();

    // DEBUG ------
    cv::Mat img_matches;
    cv::Mat _frame1, _frame2;
    images[0].copyTo(_frame1, masks[0]);
    images[1].copyTo(_frame2, masks[1]);
    drawMatches(_frame1, keypoints[0], _frame2, keypoints[1], keptMatches, img_matches,
                cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("Matches", img_matches);
    // DEBUG ------

    return result;
}

void Matcher::detectKeypointsAndExtractDescriptors(const std::vector<cv::Mat> &images,
                                                   const std::vector<cv::Mat> &masks) {
    double t0 = cv::getTickCount();

    cv::xfeatures2d::SURF detector(600);
    cv::xfeatures2d::SURF extractor;

    keypoints.resize(images.size());
    descriptors.resize(images.size());

    // FIXME -- CLEAR???
    for (int i = 0; i < images.size(); i++) {
        detector.detect(images[i], keypoints[i], masks[i]);
        extractor.compute(images[i], keypoints[i], descriptors[i]);
    }

    t0 = ((double) cv::getTickCount() - t0) / cv::getTickFrequency();
    std::cout << "-- SURF detection and extraction done in " << t0 << "s" << std::endl;
    std::cout.flush();
}

std::vector<cv::DMatch> Matcher::matchDescriptors() {
    cv::FlannBasedMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors[0], descriptors[1], matches);

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
//    std::vector<DMatch> good_matches;
//    std::cout << "Matches before distnace selection: " << matches.size() << std::endl;
//    for (int i = 0; i < descriptors[0].rows; i++) {
//        if (matches[i].distance <= max(2 * min_dist, 0.35)) {
//            good_matches.push_back(matches[i]);
//        }
//    }
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

        if (fabs(res.at<double>(0)) < threshold) {
            keptMatches.push_back(matches[i]);
            matchingPoints.push_back(std::make_pair(points1[i], points2[i]));
        }
    }

    std::cout << "- " << keptMatches.size() << "/" << matches.size() << " has been kept by F." << std::endl;

    return matchingPoints;
}
