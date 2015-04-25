#include <opencv2/features2d.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include "MyMatcher.h"
#include "SURFFeatureExtractor.h"

using namespace cv;
using namespace std;

vector<pair<Point2f, Point2f>> MyMatcher::match(SURFFeatureExtractor extractor, vector<DMatch> &resultMatches, Mat F) {

    // 2. Match the two image descriptors
    BFMatcher matcher(NORM_L2);

    // from image 1 to image 2
    // based on k nearest neighbours (with k=2)
    std::vector<std::vector<cv::DMatch>> matches1;
    matcher.knnMatch(extractor.descriptors[0], extractor.descriptors[1],
                     matches1, // vector of matches (up to 2 per entry)
                     2);

    // return 2 nearest neighbours
    // from image 2 to image 1
    // based on k nearest neighbours (with k=2)
    std::vector<std::vector<cv::DMatch>> matches2;
    matcher.knnMatch(extractor.descriptors[1], extractor.descriptors[0],
                     matches2, // vector of matches (up to 2 per entry)
                     2);

// 3. Remove matches for which NN ratio is
// > than threshold
// clean image 1 -> image 2 matches
    int removed = ratioTest(matches1);
    removed = ratioTest(matches2);

    // 4. Remove non-symmetrical matches

    std::vector<cv::DMatch> symMatches;
    symmetryTest(matches1, matches2, symMatches);

    return filterWithF(extractor.keypoints, symMatches, resultMatches, F);
}

int MyMatcher::ratioTest(std::vector<std::vector<cv::DMatch>> matches) {
    int removed = 0;
    for (std::vector<std::vector<cv::DMatch>>::iterator matchIterator = matches.begin();
         matchIterator != matches.end(); ++matchIterator) {
        if (matchIterator->size() > 1) {
            if ((*matchIterator)[0].distance / (*matchIterator)[1].distance > ratio) {
                matchIterator->clear(); // remove match
                removed++;
            }
        } else { // does not have 2 neighbours
            matchIterator->clear(); // remove match
            removed++;
        }
    }
    return removed;
}

void MyMatcher::symmetryTest(const std::vector<std::vector<cv::DMatch>> &matches1,
                             const std::vector<std::vector<cv::DMatch>> &matches2,
                             std::vector<cv::DMatch> &symMatches) {
// for all matches image 1 -> image 2
    for (std::vector<std::vector<cv::DMatch>>::
         const_iterator matchIterator1 = matches1.begin();
         matchIterator1 != matches1.end(); ++matchIterator1) {
// ignore deleted matches
        if (matchIterator1->size() < 2)
            continue;
// for all matches image 2 -> image 1
        for (std::vector<std::vector<cv::DMatch>>::
             const_iterator matchIterator2 = matches2.begin();
             matchIterator2 != matches2.end();
             ++matchIterator2) {
// ignore deleted matches
            if (matchIterator2->size() < 2)
                continue;
// Match symmetry test
            if ((*matchIterator1)[0].queryIdx == (*matchIterator2)[0].trainIdx &&
                (*matchIterator2)[0].queryIdx == (*matchIterator1)[0].trainIdx) {
// add symmetrical match
                symMatches.push_back(
                        cv::DMatch((*matchIterator1)[0].queryIdx,
                                   (*matchIterator1)[0].trainIdx,
                                   (*matchIterator1)[0].distance));
                break; // next match in image 1 -> image 2
            }
        }
    }
}

std::vector<std::pair<cv::Point2f, cv::Point2f>> MyMatcher::filterWithF(
        const std::vector<std::vector<cv::KeyPoint>> &keypoints,
        const std::vector<cv::DMatch> &before, std::vector<cv::DMatch> &after, Mat F) {

    std::vector<cv::Point2f> points1, points2;
    for (std::vector<cv::DMatch>::const_iterator it = before.begin(); it != before.end(); ++it) {
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
    undistortPoints(points1, u_points1, camera1->cameraMatrix, camera1->distCoeffs, Mat(), camera1->cameraMatrix);
    undistortPoints(points2, u_points2, camera2->cameraMatrix, camera2->distCoeffs, Mat(), camera2->cameraMatrix);

    for (int i = 0; i < u_points1.size(); i++) {
        Mat res = Mat(Matx13d(u_points2[i].x, u_points2[i].y, 1)) * F * Mat(Matx31d(u_points1[i].x, u_points1[i].y, 1));

        if (abs(res.at<double>(0)) < treshold) {
            after.push_back(before[i]);
            matchingPoints.push_back(make_pair(points1[i], points2[i]));
        }
    }

    return matchingPoints;
}
