cmake_minimum_required(VERSION 3.0)

project(Final)

find_package(OpenCV 3.0.0 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(PCL 1.7.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(CLASSES calibration/StereoCalibration.cpp camera/Camera.cpp camera/RealCamera.cpp camera/DummyCamera.cpp camera/StereoCamera.cpp calibration/Calibration.cpp calibration/CameraPoseCalculator.cpp camera/CameraPose.cpp OpenGLRenderer.cpp ObjectSelector.cpp optical_flow/OFReconstruction.cpp Triangulation.cpp Common.cpp optical_flow/OpticalFlowCalculator.cpp optical_flow/TemporalOpticalFlowCalculator.cpp optical_flow/SpatialOpticalFlowCalculator.cpp MultiView.cpp PclVisualization.cpp)

add_executable(Final main.cpp ${CLASSES})
target_compile_features(Final PRIVATE cxx_range_for)
target_link_libraries(Final ${OPENGL_LIBRARIES} ${OpenCV_LIBS} ${GLUT_LIBRARIES} v4l2 ${PCL_COMMON_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES})

add_executable(Diploma diploma.cpp ${CLASSES})
target_compile_features(Diploma PRIVATE cxx_range_for)
target_link_libraries(Diploma ${OPENGL_LIBRARIES} ${OpenCV_LIBS} ${GLUT_LIBRARIES} v4l2 ${PCL_COMMON_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES})

#add_executable(StereoCalibration calibrate.cpp ${CLASSES})
#target_compile_features(StereoCalibration PRIVATE cxx_range_for)
#target_link_libraries(StereoCalibration ${OPENGL_LIBRARIES} ${OpenCV_LIBS} ${GLUT_LIBRARIES} v4l2)

add_executable(VideoMaker video.cpp ${CLASSES})
target_compile_features(VideoMaker PRIVATE cxx_range_for)
target_link_libraries(VideoMaker ${OPENGL_LIBRARIES} ${OpenCV_LIBS} ${GLUT_LIBRARIES} v4l2 ${PCL_LIBRARIES})

add_executable(TriTest TriTest.cpp ${CLASSES} camera/VirtualCamera.cpp camera/FakeCamera.cpp)
target_compile_features(TriTest PRIVATE cxx_range_for)
target_link_libraries(TriTest ${OPENGL_LIBRARIES} ${OpenCV_LIBS} ${GLUT_LIBRARIES} v4l2 ${PCL_LIBRARIES})
