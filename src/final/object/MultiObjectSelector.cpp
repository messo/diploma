#include <map>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iomanip>

#include "MultiObjectSelector.h"
#include "../PerformanceMonitor.h"

using namespace std;
using namespace cv;

vector<Object> MultiObjectSelector::selectObjects(const std::vector<cv::Mat> &frames, const std::vector<cv::Mat> &masks) {
    PerformanceMonitor::get()->objSelectionStarted();
    double t0 = getTickCount();

    std::vector<std::vector<std::vector<cv::Point>>> contours(2);
    std::vector<Mat> newMasks(2);
    std::vector<std::vector<Blob>> blobs(2);

    for (int i = 0; i < 2; i++) {
        contours[i] = this->getContours(masks[i]);
        newMasks[i] = getTotalMask(contours[i]);

        blobs[i].clear();
        for (int j = 0; j < contours[i].size(); j++) {
            blobs[i].push_back(Blob(contours[i][j]));
        }
    }

    vector<pair<Point2f, Point2f>> points = matcher.match(frames, newMasks);

    if (points.size() < 1) {
        return vector<Object>();
    }

    std::cout << "[" << std::setw(20) << "MultiObjectSelector" << "] " << "Blob on the left: " << blobs[Camera::LEFT].size() << " right: " <<
    blobs[Camera::RIGHT].size() << endl;

    // we'll map from the bigger set
    const int from = (blobs[Camera::LEFT].size() > blobs[Camera::RIGHT].size()) ? Camera::LEFT : Camera::RIGHT;
    const int to = (blobs[Camera::LEFT].size() > blobs[Camera::RIGHT].size()) ? Camera::RIGHT : Camera::LEFT;

    std::cout << "[" << std::setw(20) << "MultiObjectSelector" << "] " << "Matching: " << from << " -> " << to << endl;

    map<int, map<int, int>> blobMatchCounters;
    map<pair<int, int>, vector<pair<Point2f, Point2f>>> pointMatchesForBlobMatches;

    for (int i = 0; i < points.size(); i++) {
        const pair<Point2f, Point2f> &matching = points[i];

        int fromBlobIdx = -1;
        for (int j = 0; j < blobs[from].size(); j++) {
            if (from == Camera::LEFT) {
                if (blobs[from][j].contains(matching.first)) {
                    fromBlobIdx = j;
                    break;
                }
            } else {
                if (blobs[from][j].contains(matching.second)) {
                    fromBlobIdx = j;
                    break;
                }
            }
        }

        if (fromBlobIdx == -1) {
            // we get some points outside the mask aswell..
            continue;
        }

        int toBlobIdx = -1;
        for (int j = 0; j < blobs[to].size(); j++) {
            if (to == Camera::LEFT) {
                if (blobs[to][j].contains(matching.first)) {
                    toBlobIdx = j;
                    break;
                }
            } else {
                if (blobs[to][j].contains(matching.second)) {
                    toBlobIdx = j;
                    break;
                }
            }
        }
        if (toBlobIdx == -1) {
            // we get some points outside the mask aswell..
            continue;
        }

//        std::cout << fromBlobIdx << " -> " << toBlobIdx << endl;

        pointMatchesForBlobMatches[make_pair(fromBlobIdx, toBlobIdx)].push_back(matching);
        blobMatchCounters[fromBlobIdx][toBlobIdx]++;
    }

    // VISUALIZATION

//    Mat vis(480, 640 * 2, CV_8UC3, Scalar(0, 0, 0));
//
//    for (int i = 0; i < 2; i++) {
//        for (int j = 0; j < blobs[i].size(); j++) {
//            cv::Moments m = moments(blobs[i][j].mask, true);
//            cv::Point center(m.m10 / m.m00, m.m01 / m.m00);
//
//            Mat blobby(480, 640, CV_8UC3, Scalar(255, 255, 255));
//
//            if (i == 1) {
//                center += Point(640, 0);
//                blobby.copyTo(vis(Rect(640, 0, 640, 480)), blobs[i][j].mask);
//            } else {
//                blobby.copyTo(vis(Rect(0, 0, 640, 480)), blobs[i][j].mask);
//            }
//
//            putText(vis, std::to_string(j), center, cv::FONT_HERSHEY_PLAIN, 1.0, Scalar(255, 255, 255), 1, LINE_AA);
//        }
//    }
//
//    for (auto it = blobMatchCounters.begin(); it != blobMatchCounters.end(); ++it) {
//        int fromIdx = it->first;
//        for (auto it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
//            int toIdx = it2->first;
//
//            cv::Moments m = moments(blobs[from][fromIdx].mask, true);
//            cv::Point fromPt(m.m10 / m.m00, m.m01 / m.m00);
//
//            m = moments(blobs[to][toIdx].mask, true);
//            cv::Point toPt(m.m10 / m.m00, m.m01 / m.m00);
//
//            if (it2->second < 100) {
//                fromPt += Point(0, 80);
//                toPt += Point(0, 80);
//            }
//
//            if (from == 0) {
//                toPt += Point(640, 0);
//            } else {
//                fromPt += Point(640, 0);
//            }
//
//            putText(vis, std::to_string(it2->second), (fromPt + toPt) / 2, cv::FONT_HERSHEY_PLAIN, 2.0, Scalar(255, 255, 255), 1, LINE_AA);
//
//            if (from == 0) {
//                cv::line(vis, fromPt, toPt, Scalar(0, 0, 255), 1, LINE_AA);
//            } else {
//                cv::line(vis, toPt, fromPt, Scalar(0, 0, 255), 1, LINE_AA);
//            }
//        }
//    }
//
//    imshow("vis", vis);

    // ---------------

    vector<pair<int, int>> blobMatches;
    map<int, vector<int>> matchesBackward;

    for (auto it = blobMatchCounters.begin(); it != blobMatchCounters.end(); ++it) {
        int fromIdx = it->first;
        std::cout << (*it).first << " --> ";

        int maxCount = 0;
        int toIdx = -1;
        for (auto it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
            int id = it2->first;
            int count = it2->second;
            if (count > maxCount) {
                maxCount = count;
                toIdx = id;
            }
            std::cout << "[" << (*it2).first << ": " << (*it2).second << "] ";
        }

        matchesBackward[toIdx].push_back(fromIdx);
        blobMatches.push_back(make_pair(fromIdx, toIdx));

        std::cout << endl;
    }

//    for (int i = 0; i < blobMatches.size(); i++) {
//        std::cout << blobMatches[i].first << " -> " << blobMatches[i].second << std::endl;
//    }

    vector<Object> objects;

    // now create the objects
    for (auto it = matchesBackward.begin(); it != matchesBackward.end(); ++it) {
        int toBlobIdx = it->first;

        const Blob &toBlob = blobs[to][toBlobIdx];
        const vector<int> &fromIds = it->second;

        vector<pair<Point2f, Point2f>> pointMatches;

        // merging the masks
        Mat fromMask(480, 640, CV_8U, Scalar(0));
        for (int i = 0; i < fromIds.size(); i++) {
            vector<pair<Point2f, Point2f>> &vec = pointMatchesForBlobMatches[make_pair(fromIds[i], toBlobIdx)];
            pointMatches.insert(pointMatches.end(), vec.begin(), vec.end());

            blobs[from][fromIds[i]].mask.copyTo(fromMask, blobs[from][fromIds[i]].mask);
        }

        if (from == Camera::LEFT) {
            objects.push_back(Object(fromMask, toBlob.mask));
        } else {
            objects.push_back(Object(toBlob.mask, fromMask));
        }

        objects.back().matches = pointMatches;
    }

    t0 = ((double) getTickCount() - t0) / getTickFrequency();
    std::cout << "[" << std::setw(20) << "MultiObjectSelector" << "] " << "MultiObject selection done in " << t0 << "s" << std::endl;
    std::cout.flush();

    PerformanceMonitor::get()->objSelectionFinished();

    return objects;
}

std::vector<std::vector<Point>> MultiObjectSelector::getContours(const cv::Mat &mask) {
    std::vector<std::vector<Point>> allContours;

    Mat img;
    mask.copyTo(img);
    findContours(img, allContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<Point>> selectedContours;

    for (int idx = 0; idx < allContours.size(); idx++) {
        const std::vector<Point> &c = allContours[idx];
        double area = fabs(contourArea(c));
        if (area > MIN_AREA) {
            selectedContours.push_back(c);
        }
    }

    return selectedContours;
}

Mat MultiObjectSelector::getTotalMask(std::vector<std::vector<cv::Point>> &contours) {
    Mat result(480, 640, CV_8U, Scalar(0));
    cv::drawContours(result, contours, -1, Scalar(255), -1, LINE_AA);
    return result;
}
