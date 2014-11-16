#include <iostream>
#include <string>

#include <libv4l2.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <fcntl.h>

#include "RealCamera.hpp"

RealCamera::RealCamera(int id) : Camera(id) {
    // open capture
    int descriptor = v4l2_open(("/dev/video" + std::to_string(id)).c_str(), O_RDWR);

    // manual exposure control
    v4l2_control c;
    c.id = V4L2_CID_EXPOSURE_AUTO;
    c.value = V4L2_EXPOSURE_MANUAL;
    if (v4l2_ioctl(descriptor, VIDIOC_S_CTRL, &c) == 0)
        std::cout << "Setting V4L2_CID_EXPOSURE_AUTO succeeded." << std::endl;

    c.id = V4L2_CID_EXPOSURE_ABSOLUTE;
    c.value = 1500;
    if (v4l2_ioctl(descriptor, VIDIOC_S_CTRL, &c) == 0)
        std::cout << "Setting V4L2_CID_EXPOSURE_ABSOLUTE succeeded." << std::endl;

    // auto priority control
    c.id = V4L2_CID_EXPOSURE_AUTO_PRIORITY;
    c.value = 1;
    if (v4l2_ioctl(descriptor, VIDIOC_S_CTRL, &c) == 0)
        std::cout << "Setting V4L2_CID_EXPOSURE_AUTO_PRIORITY succeeded." << std::endl;

    if (!cap.open(id)) {
        std::cout << "Cannot open camera #" << id << std::endl;
    }
}

bool RealCamera::read(cv::OutputArray img) {
    return cap.read(img);
}

bool RealCamera::grab() {
    return cap.grab();
}

bool RealCamera::retrieve(cv::OutputArray img) {
    return cap.retrieve(img);
}
