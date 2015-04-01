cmake_minimum_required(VERSION 3.0)

project(Final)

find_package(OpenCV 3.0.0 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(PCL 1.7.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(CLASSES Calibration.cpp Camera.cpp RealCamera.cpp DummyCamera.cpp DinoCamera.cpp StereoCamera.cpp OpenGLRenderer.cpp ObjectSelector.cpp OFReconstruction.cpp Triangulation.cpp Common.cpp OpticalFlowCalculator.cpp MultiView.cpp PclVisualization.cpp)

add_executable(Final main.cpp ${CLASSES})
target_compile_features(Final PRIVATE cxx_range_for)
target_link_libraries(Final ${OPENGL_LIBRARIES} ${OpenCV_LIBS} ${GLUT_LIBRARIES} v4l2 ${PCL_COMMON_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES})

#add_executable(Calibration calibrate.cpp ${CLASSES})
#target_compile_features(Calibration PRIVATE cxx_range_for)
#target_link_libraries(Calibration ${OPENGL_LIBRARIES} ${OpenCV_LIBS} ${GLUT_LIBRARIES} v4l2)

add_executable(VideoMaker video.cpp ${CLASSES})
target_compile_features(VideoMaker PRIVATE cxx_range_for)
target_link_libraries(VideoMaker ${OPENGL_LIBRARIES} ${OpenCV_LIBS} ${GLUT_LIBRARIES} v4l2 ${PCL_LIBRARIES})

add_executable(TriTest TriTest.cpp ${CLASSES} VirtualCamera.cpp FakeCamera.cpp)
target_compile_features(TriTest PRIVATE cxx_range_for)
target_link_libraries(TriTest ${OPENGL_LIBRARIES} ${OpenCV_LIBS} ${GLUT_LIBRARIES} v4l2 ${PCL_LIBRARIES})
