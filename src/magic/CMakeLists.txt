cmake_minimum_required(VERSION 2.8.4)
project(Magic)

find_package(OpenCV 3.0.0 REQUIRED)
find_package(OpenGL REQUIRED)

set(CLASSES Calibration.cpp StereoCamera.cpp RealStereoCamera.cpp)

add_executable(Magic main.cpp ${CLASSES})
target_link_libraries(Magic ${OPENGL_LIBRARIES} ${OpenCV_LIBS})

add_executable(Calibration calibrate.cpp ${CLASSES})
target_link_libraries(Calibration ${OPENGL_LIBRARIES} ${OpenCV_LIBS})
