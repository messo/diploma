#include "Calibration.hpp"

using namespace std;
using namespace cv;

Calibration::Calibration(const string &intrinsics, const string &extrinsics) :
        imageSize(640, 480) {

    cameraMatrix[0] = Mat::eye(3, 3, CV_64F);
    cameraMatrix[1] = Mat::eye(3, 3, CV_64F);

    // load intrinsic parameters
    FileStorage fs(intrinsics, FileStorage::READ);
    if (fs.isOpened()) {
        fs["M1"] >> cameraMatrix[0];
        fs["D1"] >> distCoeffs[0];
        fs["M2"] >> cameraMatrix[1];
        fs["D2"] >> distCoeffs[1];
        fs.release();
    } else
        cout << "Error: can not load the intrinsic parameters\n";

    Mat R1, R2;

    fs.open(extrinsics, FileStorage::READ);
    if (fs.isOpened()) {
        fs["R"] >> R;
        fs["T"] >> T;
        fs["E"] >> E;
        fs["F"] >> F;
        fs["R1"] >> R1;
        fs["R2"] >> R2;
        fs["P1"] >> P1;
        fs["P2"] >> P2;
        fs["Q"] >> Q;
        fs["validRoiLeft"] >> validRoiLeft;
        fs["validRoiRight"] >> validRoiRight;
        fs.release();
    } else
        cout << "Error: can not load the intrinsic parameters\n";

    //Precompute maps for cv::remap()
    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize,
            CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize,
            CV_16SC2, rmap[1][0], rmap[1][1]);
}

void Calibration::acquireFrames(StereoCamera &stereoCamera) {
    images.push_back(stereoCamera.getLeft());
    images.push_back(stereoCamera.getRight());
    cout << "Frames aquired. \n";
    cout << "Current number of frames(2x): " << images.size() << "\n";
    cout.flush();
}

void Calibration::calibrate() {
    const bool displayCorners = true;
    const int maxScale = 2;
    const float squareSize = 1.f;
    const Size boardSize(9, 6);
    const bool showRectified = true;
    const bool useCalibrated = true;

    cout << "Calibration started... \n";
    cout.flush();

    vector<vector<Point2f> > imagePoints[2];
    vector<vector<Point3f> > objectPoints;
    Size imageSize;

    uint i, j, k, nimages = (uint) images.size() / 2;

    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    vector<Mat> goodImageList;

    for (i = j = 0; i < nimages; i++) {
        for (k = 0; k < 2; k++) {
            Mat cimg = images[i * 2 + k];
            Mat img;
            cvtColor(cimg, img, COLOR_BGR2GRAY);
            if (img.empty())
                break;
            if (imageSize == Size())
                imageSize = img.size();
            bool found = false;
            vector<Point2f> &corners = imagePoints[k][j];
            for (int scale = 1; scale <= maxScale; scale++) {
                Mat timg;
                if (scale == 1)
                    timg = img;
                else
                    resize(img, timg, Size(), scale, scale);
                found = findChessboardCorners(timg, boardSize, corners,
                        CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
                if (found) {
                    if (scale > 1) {
                        Mat cornersMat(corners);
                        cornersMat *= 1. / scale;
                    }
                    break;
                }
            }
            if (displayCorners) {
                Mat cimg, cimg1;
                cvtColor(img, cimg, COLOR_GRAY2BGR);
                drawChessboardCorners(cimg, boardSize, corners, found);
                double sf = 640. / MAX(img.rows, img.cols);
                resize(cimg, cimg1, Size(), sf, sf);
                imshow("corners", cimg1);
                char c = (char) waitKey(500);
                if (c == 27 || c == 'q' || c == 'Q') //Allow ESC to quit
                    exit(-1);
            }
            else
                putchar('.');
            if (!found)
                break;
            cornerSubPix(img, corners, Size(11, 11), Size(-1, -1),
                    TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS,
                            30, 0.01));
        }
        if (k == 2) {
            goodImageList.push_back(images[i * 2]);
            goodImageList.push_back(images[i * 2 + 1]);
            j++;
        }
    }
    cout << j << " pairs have been successfully detected.\n";
    nimages = j;
    if (nimages < 2) {
        cout << "Error: too little pairs to run the calibration\n";
        return;
    }

    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    objectPoints.resize(nimages);

    for (i = 0; i < nimages; i++) {
        for (j = 0; j < boardSize.height; j++)
            for (k = 0; k < boardSize.width; k++)
                objectPoints[i].push_back(Point3f(j * squareSize, k * squareSize, 0));
    }

    cout << "Running stereo calibration ...\n";

    cameraMatrix[0] = Mat::eye(3, 3, CV_64F);
    cameraMatrix[1] = Mat::eye(3, 3, CV_64F);

    double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
            cameraMatrix[0], distCoeffs[0],
            cameraMatrix[1], distCoeffs[1],
            imageSize, R, T, E, F,
            CALIB_FIX_ASPECT_RATIO +
                    CALIB_ZERO_TANGENT_DIST +
                    CALIB_SAME_FOCAL_LENGTH +
                    CALIB_RATIONAL_MODEL +
                    CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5,
            TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 100, 1e-5)
    );
    cout << "done with RMS error=" << rms << endl;

// CALIBRATION QUALITY CHECK
// because the output fundamental matrix implicitly
// includes all the output information,
// we can check the quality of calibration using the
// epipolar geometry constraint: m2^t*F*m1=0
    double err = 0;
    int npoints = 0;
    vector<Vec3f> lines[2];
    for (i = 0; i < nimages; i++) {
        int npt = (int) imagePoints[0][i].size();
        Mat imgpt[2];
        for (k = 0; k < 2; k++) {
            imgpt[k] = Mat(imagePoints[k][i]);
            undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
            computeCorrespondEpilines(imgpt[k], k + 1, F, lines[k]);
        }
        for (j = 0; j < npt; j++) {
            double errij = fabs(imagePoints[0][i][j].x * lines[1][j][0] +
                    imagePoints[0][i][j].y * lines[1][j][1] + lines[1][j][2]) +
                    fabs(imagePoints[1][i][j].x * lines[0][j][0] +
                            imagePoints[1][i][j].y * lines[0][j][1] + lines[0][j][2]);
            err += errij;
        }
        npoints += npt;
    }
    cout << "average reprojection err = " << err / npoints << endl;

    // save intrinsic parameters
    FileStorage fs("intrinsics.yml", FileStorage::WRITE);
    if (fs.isOpened()) {
        fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
                "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
        fs.release();
    }
    else
        cout << "Error: can not save the intrinsic parameters\n";

    Mat R1, R2;
    Rect validRoiLeft, validRoiRight;

    stereoRectify(cameraMatrix[0], distCoeffs[0],
            cameraMatrix[1], distCoeffs[1],
            imageSize, R, T, R1, R2, P1, P2, Q,
            CALIB_ZERO_DISPARITY, 1, imageSize, &validRoiLeft, &validRoiRight);

    // OpenCV can handle left-right
    // or up-down camera arrangements
    bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));

    // COMPUTE AND DISPLAY RECTIFICATION
    if (!showRectified)
        return;

    // IF BY CALIBRATED (BOUGUET'S METHOD)
    if (useCalibrated) {
        // we already computed everything
    }

        // OR ELSE HARTLEY'S METHOD
    else
        // use intrinsic parameters of each camera, but
        // compute the rectification transformation directly
        // from the fundamental matrix
    {
        vector<Point2f> allimgpt[2];
        for (k = 0; k < 2; k++) {
            for (i = 0; i < nimages; i++)
                std::copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), back_inserter(allimgpt[k]));
        }
        F = findFundamentalMat(Mat(allimgpt[0]), Mat(allimgpt[1]), FM_8POINT, 0, 0);
        Mat H1, H2;
        stereoRectifyUncalibrated(Mat(allimgpt[0]), Mat(allimgpt[1]), F, imageSize, H1, H2, 3);

        R1 = cameraMatrix[0].inv() * H1 * cameraMatrix[0];
        R2 = cameraMatrix[1].inv() * H2 * cameraMatrix[1];
        P1 = cameraMatrix[0];
        P2 = cameraMatrix[1];
    }

    fs.open("extrinsics.yml", FileStorage::WRITE);
    if (fs.isOpened()) {
        fs << "R" << R << "T" << T << "E" << E << "F" << F << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q << "validRoiLeft" << validRoiLeft << "validRoiRight" << validRoiRight;
        fs.release();
    }
    else
        cout << "Error: can not save the intrinsic parameters\n";

    //Precompute maps for cv::remap()
    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

    Mat canvas;
    double sf;
    int w, h;
    if (!isVerticalStereo) {
        sf = 600. / MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width * sf);
        h = cvRound(imageSize.height * sf);
        canvas.create(h, w * 2, CV_8UC3);
    }
    else {
        sf = 300. / MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width * sf);
        h = cvRound(imageSize.height * sf);
        canvas.create(h * 2, w, CV_8UC3);
    }

    for (i = 0; i < nimages; i++) {
        for (k = 0; k < 2; k++) {
            Mat img = goodImageList[i * 2 + k], cimg;
            remap(img, cimg, rmap[k][0], rmap[k][1], INTER_LINEAR);
            Mat canvasPart = !isVerticalStereo ? canvas(Rect(w * k, 0, w, h)) : canvas(Rect(0, h * k, w, h));
            resize(cimg, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
            if (useCalibrated) {
                Rect validRoi = (k == 0) ? validRoiLeft : validRoiRight;
                Rect vroi(cvRound(validRoi.x * sf), cvRound(validRoi.y * sf),
                        cvRound(validRoi.width * sf), cvRound(validRoi.height * sf));
                rectangle(canvasPart, vroi, Scalar(0, 0, 255), 3, 8);
            }
        }

        if (!isVerticalStereo)
            for (j = 0; j < canvas.rows; j += 16)
                line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
        else
            for (j = 0; j < canvas.cols; j += 16)
                line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
        imshow("rectified", canvas);
        char c = (char) waitKey();
        if (c == ' ') {
            imwrite("output.jpg", canvas);
        }
        if (c == 27 || c == 'q' || c == 'Q')
            break;
    }
}
