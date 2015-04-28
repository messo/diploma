#include "Object.h"

Object::Object(cv::Mat left, cv::Mat right) : masks(2) {
    masks[0] = left;
    masks[1] = right;
}
