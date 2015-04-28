#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui.hpp>
#include "../Common.h"
#include "MultiObjectOpticalFlowCalculator.h"

using namespace cv;
using namespace std;


bool MultiObjectOpticalFlowCalculator::feed(std::vector<Mat> &frames, const Object &object) {

    Point translations[2];

    double t0 = getTickCount();

#pragma omp parallel for
    for (int i = 0; i < 2; i++) {
        if (frames[i].channels() != 1) {
            cvtColor(frames[i], this->frames[i], COLOR_BGR2GRAY);
        } else {
            this->frames[i] = frames[i].clone();
        }
        this->masks[i] = object.masks[i].clone();

        // move the frame2 and mask around a bit, so the main object's movement is not that big -- so OF will be okay
        //translations[i] = moveToTheCenter(this->frames[i], this->masks[i]);

//         this->texturedRegions[i] = masks[i].clone();
        this->calcTexturedRegions(this->frames[i], this->masks[i], this->texturedRegions[i]);
    }

    // SHIFTING....
    vector<Point2f> vectors;
    for (int i = 0; i < object.matches.size(); i++) {
        vectors.push_back(object.matches[i].second - object.matches[i].first);
    }
    Point2f v = magicVector(vectors);
    //Point2f v(0, 0);

    Point optimalShift(cvRound(v.x), cvRound(v.y));

    shiftFrame(1, -optimalShift);

//    Rect unified = boundingRect(this->masks[0]) | boundingRect(this->masks[1]);
//    Mat merged = mergeImagesVertically(this->frames[0](unified), this->frames[1](unified));
//    imwrite("/media/balint/Data/Linux/diploma/after_shift.png", merged);

    imshow("frame1", this->frames[0]);
    imshow("frame2", this->frames[1]);

    // --------------------

//    Rect br0 = boundingRect(this->masks[0]);
//    Rect br1 = boundingRect(this->masks[1]);
//    Rect uni = br0 | br1;
//    Mat frame0 = frames[0].clone();
//    Mat frame1 = frames[1].clone();
//    rectangle(frame0, uni.tl(), uni.br(), Scalar(0, 0, 255), 2, LINE_AA);
//    rectangle(frame0, br0.tl(), br0.br(), Scalar(0, 255, 255), 1, LINE_AA);
//    rectangle(frame1, uni.tl(), uni.br(), Scalar(0, 0, 255), 2, LINE_AA);
//    rectangle(frame1, br1.tl(), br1.br(), Scalar(0, 255, 255), 1, LINE_AA);
//    imwrite("/media/balint/Data/Linux/diploma/of_img_left_framed.png", frame0);
//    imwrite("/media/balint/Data/Linux/diploma/of_img_right_framed.png", frame1);


    // texturazott cuccok...
//    Rect br0 = boundingRect(this->masks[0]);
//    Rect br1 = boundingRect(this->masks[1]);
//    Mat textures = mergeImages(this->texturedRegions[0](br0), this->texturedRegions[1](br1));
//    imshow("textures", textures);
//    imwrite("/media/balint/Data/Linux/diploma/textures.png", textures);


    t0 = ((double) getTickCount() - t0) / getTickFrequency();
    std::cout << "Feed init done in " << t0 << "s" << std::endl;
    std::cout.flush();

    this->calcOpticalFlow(translations[1]);

//    this->visualizeMatches(points1, points2);

    // move the points to their original location
    for (int i = 0; i < points1.size(); i++) {
        points2[i] += Point2f(optimalShift);
    }

//    this->visualizeMatches(frames[0], points1, frames[1], points2);

    return true;
}

double MultiObjectOpticalFlowCalculator::calcOpticalFlow(cv::Point &translation) {
    Mat flows[2];
    Mat _flows[2];

    // find a common bounding rect to simplify the OF
    Rect rect0(boundingRect(masks[0]));
    Rect rect1(boundingRect(masks[1]));
    Rect unified(rect0 | rect1);

#pragma omp parallel for
    for (int i = 0; i < 2; i++) {
        //std::cout << "THREAD: " << omp_get_thread_num() << std::endl;
        calcOpticalFlowFarneback(frames[i](unified), frames[(i + 1) % 2](unified), _flows[i], 0.75, 6, 21, 10, 7, 1.5,
                                 OPTFLOW_FARNEBACK_GAUSSIAN);

        // put the flows back into the full matrix
        flows[i] = Mat::zeros(frames[0].rows, frames[0].cols, CV_32FC2);
        _flows[i].copyTo(flows[i](unified));
    }

//    this->visualizeOpticalFlow(frames[0], masks[0], frames[1], masks[1], flows[0], "flow12");
//    this->visualizeOpticalFlow(frames[1], masks[1], frames[0], masks[0], flows[1], "flow21");

    // collecting matching points using optical flow
    collectMatchingPoints(flows[0], flows[1], unified, points1, points2);

    std::cout << points1.size() << std::endl;
    std::cout.flush();

    Point2f avgMovement(this->calcAverageMovement(points1, points2));

    double length = norm(avgMovement);
    cout << avgMovement << ": " << length << endl;
    cout.flush();

//    if(points1.size() < 10000) {
//    if (length > 5.0) {
//        // we consider this as "too big" displacement, so we shift the current image further, and do this again
//
//        Point newTranslation(-avgMovement * 0.75);
//
//        Rect currentBoundingRect(boundingRect(masks[1]));
//
//        // translate the image
//        Mat _currentFrame;
//        frames[1].copyTo(_currentFrame);
//        shiftImage(_currentFrame, currentBoundingRect, newTranslation, frames[1]);
//        // translate the mask
//        Mat _currentMask;
//        masks[1].copyTo(_currentMask);
//        shiftImage(_currentMask, currentBoundingRect, newTranslation, masks[1]);
//        // translate the textured regions
//        Mat _currentTexturedRegions;
//        texturedRegions[1].copyTo(_currentTexturedRegions);
//        shiftImage(_currentTexturedRegions, currentBoundingRect, newTranslation, texturedRegions[1]);
//
//        translation += newTranslation;
//
//        length = this->calcOpticalFlow(translation);
//    } else {
    visualizeMatchesROI(frames[0], points1, frames[1], points2);
//    }

    return 2.0; //length;
}

void MultiObjectOpticalFlowCalculator::collectMatchingPoints(const cv::Mat &flow, const cv::Mat &backFlow,
                                                             const cv::Rect &roi, std::vector<cv::Point2f> &points1,
                                                             std::vector<cv::Point2f> &points2) {
    points1.clear();
    points2.clear();

//    double t0 = getTickCount();

    Rect imageArea(Point(0, 0), masks[1].size());

    for (int y = roi.tl().y; y <= roi.br().y; y++) {
        for (int x = roi.tl().x; x <= roi.br().x; x++) {

            // from + fwd = to
            Point from(x, y);
            const Point2f &fwd = flow.at<Point2f>(from);
            Point to(cvRound(x + fwd.x), cvRound(y + fwd.y));

            // check if we are still in the mask, and the movement is not zero
            if ((masks[0].at<uchar>(from) == 255) && to.inside(imageArea) && (masks[1].at<uchar>(to) == 255)) {

                // to + back = backTo
                const Point2f &back = backFlow.at<Point2f>(to);
                Point backTo(cvRound(to.x + back.x), cvRound(to.y + back.y));

                Point diff2(backTo.x - from.x, backTo.y - from.y);

                // only use if the backflow points almost there
                if (diff2.dot(diff2) <= 1) {
                    if (texturedRegions[0].at<char>(from) && texturedRegions[1].at<char>(to)) {
                        points1.push_back(Point2f(from));
                        points2.push_back(Point2f(x + fwd.x, y + fwd.y));
                    }
                }
            }
        }
    }
}

