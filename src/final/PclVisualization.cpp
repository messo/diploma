#include "PclVisualization.h"
#include <pcl/surface/texture_mapping.h>

using namespace pcl;

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void *viewer_void) {
    boost::shared_ptr<visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *> (viewer_void);
    if (event.getKeySym() == "e" && event.keyDown()) {
        viewer->spinOnce();
    }
}

PclVisualization::PclVisualization() : visu("pcl visualization") {

    visu.addCoordinateSystem(1.0, "global");

//    Clipping plane [near,far] 0.024828, 24.828
//    Focal point [x,y,z] -0.457301, 0.33872, 8.81649
//    Position [x,y,z] 1.87121, -3.00263, -4.96935
//    View up [x,y,z] -0.396604, -0.905247, 0.152421

    visualization::Camera cam;
    cam.pos[0] = 0.0494508;
    cam.pos[1] = -0.388455;
    cam.pos[2] = 5.81629;

    cam.view[0] = -0.3966;
    cam.view[1] = -0.905247;
    cam.view[2] = 0.152421;

    cam.focal[0] = -0.457301;
    cam.focal[1] = 0.33872;
    cam.focal[2] = 8.81649;

    cam.clip[0] = 0.008923;
    cam.clip[1] = 8.923;

    cam.fovy = M_PI_2;

    cam.window_size[0] = 640;
    cam.window_size[1] = 480;

    visu.setCameraParameters(cam);

    // visu.registerKeyboardCallback(keyboardEventOccurred, (void *) &visu);
}

void PclVisualization::addCamera(cv::Ptr<Camera> cam, const cv::Matx44d &P, long frameId) {
    Eigen::Affine3d pose;

//    pose(0, 3) = P(0, 3); // TX
//    pose(1, 3) = P(1, 3); // TY
//    pose(2, 3) = P(2, 3); // TZ
//
//    pose(0, 0) = P(0, 0);
//    pose(0, 1) = P(0, 1);
//    pose(0, 2) = P(0, 2);
//
//    pose(1, 0) = P(1, 0);
//    pose(1, 1) = P(1, 1);
//    pose(1, 2) = P(1, 2);
//
//    pose(2, 0) = P(2, 0);
//    pose(2, 1) = P(2, 1);
//    pose(2, 2) = P(2, 2);
//
//    pose(3, 0) = 0.0;
//    pose(3, 1) = 0.0;
//    pose(3, 2) = 0.0;
//    pose(3, 3) = 1.0; //Scale

    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            pose(r, c) = P(r, c);
        }
    }

    // add a visual for each camera at the correct pose
    double focal = cam->getFocalLength();
    double height = cam->getHeight();
    double width = cam->getWidth();

    // create a 5-point visual for each camera
    PointXYZ p1, p2, p3, p4, p5;
    p1.x = 0;
    p1.y = 0;
    p1.z = 0;
    double angleX = RAD2DEG(2.0 * atan(width / (2.0 * focal)));
    double angleY = RAD2DEG(2.0 * atan(height / (2.0 * focal)));
    double dist = -0.75;
    double minX, minY, maxX, maxY;
    maxX = dist * tan(atan(width / (2.0 * focal)));
    minX = -maxX;
    maxY = dist * tan(atan(height / (2.0 * focal)));
    minY = -maxY;
    p2.x = minX;
    p2.y = minY;
    p2.z = dist;
    p3.x = maxX;
    p3.y = minY;
    p3.z = dist;
    p4.x = maxX;
    p4.y = maxY;
    p4.z = dist;
    p5.x = minX;
    p5.y = maxY;
    p5.z = dist;
    p1 = transformPoint(p1, pose);
    p2 = transformPoint(p2, pose);
    p3 = transformPoint(p3, pose);
    p4 = transformPoint(p4, pose);
    p5 = transformPoint(p5, pose);

    //std::stringstream ss;
    //ss << "Cam #" << i + 1;
    //visu.addText3D("Cam", p1, 0.1, 1.0, 1.0, 1.0, "ss");

    visu.addLine(p1, p2, "l1_" + std::to_string(frameId));
    visu.addLine(p1, p3, "l2_" + std::to_string(frameId));
    visu.addLine(p1, p4, "l3_" + std::to_string(frameId));
    visu.addLine(p1, p5, "l4_" + std::to_string(frameId));
    visu.addLine(p2, p5, "l5_" + std::to_string(frameId));
    visu.addLine(p5, p4, "l6_" + std::to_string(frameId));
    visu.addLine(p4, p3, "l7_" + std::to_string(frameId));
    visu.addLine(p3, p2, "l8_" + std::to_string(frameId));
}

void PclVisualization::addPointCloud(const std::vector<CloudPoint> &points, long frameId) {

    PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>());

    for (int i = 0; i < points.size(); i++) {
        const cv::Point3d &p = points[i].pt;
        cloud->push_back(PointXYZ(p.x, p.y, p.z));
    }

    visualization::PointCloudColorHandlerGenericField<PointXYZ> color_handler(cloud, "z");

    visu.addPointCloud(cloud, color_handler, std::to_string(frameId));
    visu.spin();
}

void PclVisualization::init() {
    visu.removeAllPointClouds();
    visu.removeAllShapes();
}

void PclVisualization::addChessboard() {
    PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>());

    for (int y = 0; y < 6; y++) {
        for (int x = 0; x < 9; x++) {
            cloud->push_back(PointXYZ(y, x, 0));
        }
    }

    visualization::PointCloudColorHandlerGenericField<PointXYZ> color_handler(cloud, "z");

    visu.addPointCloud(cloud, color_handler, "chessboard");
}
