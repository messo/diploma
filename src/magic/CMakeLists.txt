cmake_minimum_required(VERSION 2.8.4)
project(Magic)

find_package(OpenCV 3.0.0 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

list(APPEND CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")

set(CLASSES Calibration.cpp Camera.hpp RealCamera.cpp DummyCamera.cpp StereoCamera.cpp OpenGLRenderer.cpp)

add_executable(Magic main.cpp ${CLASSES})
target_link_libraries(Magic ${OPENGL_LIBRARIES} ${OpenCV_LIBS} ${GLUT_LIBRARIES} v4l2)

add_executable(Calibration calibrate.cpp ${CLASSES})
target_link_libraries(Calibration ${OPENGL_LIBRARIES} ${OpenCV_LIBS} v4l2)

add_executable(VideoMaker video.cpp ${CLASSES})
target_link_libraries(VideoMaker ${OPENGL_LIBRARIES} ${OpenCV_LIBS} v4l2)
