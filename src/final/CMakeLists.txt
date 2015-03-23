cmake_minimum_required(VERSION 2.8.4)
project(Final)

find_package(OpenCV 3.0.0 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

list(APPEND CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")

set(CLASSES Calibration.cpp Camera.hpp RealCamera.cpp DummyCamera.cpp DinoCamera.cpp StereoCamera.cpp OpenGLRenderer.cpp)

add_executable(Final main.cpp ${CLASSES} ObjectSelector.cpp ObjectSelector.hpp Camera.cpp OFReconstruction.h OFReconstruction.cpp Triangulation.h Triangulation.cpp Common.h Common.cpp OpticalFlowCalculator.cpp OpticalFlowCalculator.h)
target_link_libraries(Final ${OPENGL_LIBRARIES} ${OpenCV_LIBS} ${GLUT_LIBRARIES} v4l2)

add_executable(Calibration calibrate.cpp ${CLASSES})
target_link_libraries(Calibration ${OPENGL_LIBRARIES} ${OpenCV_LIBS} ${GLUT_LIBRARIES} v4l2)

add_executable(VideoMaker video.cpp ${CLASSES})
target_link_libraries(VideoMaker ${OPENGL_LIBRARIES} ${OpenCV_LIBS} ${GLUT_LIBRARIES} v4l2)

add_executable(TriTest TriTest.cpp ${CLASSES} ObjectSelector.cpp ObjectSelector.hpp Camera.cpp OFReconstruction.h OFReconstruction.cpp Triangulation.h Triangulation.cpp Common.h Common.cpp VirtualCamera.h VirtualCamera.cpp FakeCamera.h FakeCamera.cpp)
target_link_libraries(TriTest ${OPENGL_LIBRARIES} ${OpenCV_LIBS} ${GLUT_LIBRARIES} v4l2)
