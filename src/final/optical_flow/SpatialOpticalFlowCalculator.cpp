#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui.hpp>
#include "../Common.h"
#include "SpatialOpticalFlowCalculator.h"

using namespace cv;
using namespace std;


bool SpatialOpticalFlowCalculator::feed(std::vector<cv::Mat> &frames, const Object &object) {

    double t0 = getTickCount();

#pragma omp parallel for
    for (int i = 0; i < 2; i++) {
        if (frames[i].channels() != 1) {
            cvtColor(frames[i], this->frames[i], COLOR_BGR2GRAY);
        } else {
            this->frames[i] = frames[i].clone();
        }
        this->masks[i] = object.masks[i]; //.clone();

//         this->texturedRegions[i] = masks[i].clone();
        this->calcTexturedRegions(this->frames[i], this->masks[i], this->texturedRegions[i]);
    }

    // SHIFT
    Point optimalShift;
    if (object.matches.size() >= 2) {
        vector<Point2f> vectors;
        for (int i = 0; i < object.matches.size(); i++) {
            vectors.push_back(object.matches[i].second - object.matches[i].first);
        }
        Point2f v = magicVector(vectors);
        optimalShift = Point(cvRound(v.x), cvRound(v.y));

        std::cout << "SHIFT: " << optimalShift << endl;
        shiftFrame(1, -optimalShift);
    } else {
        std::cerr << "Not enough matches, no shifting!" << endl;
    }

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

    this->calcOpticalFlow();

//    this->visualizeMatches(points1, points2);

    // move the points to their original location
    for (int i = 0; i < points1.size(); i++) {
        points2[i] += Point2f(optimalShift);
    }

//    this->visualizeMatches(frames[0], points1, frames[1], points2);

    return true;
}

