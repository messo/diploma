#pragma once

#include "Camera.hpp"

class FakeCamera : public Camera {

public:
    FakeCamera(int id);

    virtual bool read(cv::_OutputArray const &img);

    virtual bool grab();

    virtual bool retrieve(cv::_OutputArray const &img);
};