cmake_minimum_required(VERSION 3.0)

project(Final)

find_package(OpenCV 3.0.0 REQUIRED)
find_package(OpenMP)

include_directories(/usr/include/eigen3)

#find_package(OpenGL REQUIRED)
#find_package(GLUT REQUIRED)

#find_package(PCL 1.7.2 REQUIRED)
#include_directories(${PCL_INCLUDE_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(CLASSES camera/Camera.cpp camera/RealCamera.cpp camera/DummyCamera.cpp
            calibration/Calibration.cpp calibration/CameraPoseCalculator.cpp camera/CameraPose.cpp
            mask/MOG2ForegroundMaskCalculator.cpp mask/OFForegroundMaskCalculator.cpp
            object/Matcher.cpp object/ObjectSelector.cpp object/SingleObjectSelector.cpp object/MultiObjectSelector.cpp object/Blob.cpp object/Object.cpp
            optical_flow/OpticalFlowCalculator.cpp
            Common.cpp
            Triangulator.cpp
            Visualization.cpp
)

add_executable(Diploma diploma.cpp ${CLASSES} locking.h FPSCounter.cpp)
target_compile_features(Diploma PRIVATE cxx_range_for)
target_link_libraries(Diploma ${OpenCV_LIBS} v4l2)

add_executable(ForReport for_report.cpp ${CLASSES} optical_flow/ReportOpticalFlowCalculator.cpp)
target_compile_features(ForReport PRIVATE cxx_range_for)
target_link_libraries(ForReport ${OpenCV_LIBS} v4l2)

add_executable(MultiObj multi_obj.cpp ${CLASSES})
target_compile_features(MultiObj PRIVATE cxx_range_for)
target_link_libraries(MultiObj ${OpenCV_LIBS} v4l2)


#add_executable(Final __unused/sfm.cpp ${CLASSES})
#target_compile_features(Final PRIVATE cxx_range_for)
#target_link_libraries(Final ${OpenCV_LIBS} v4l2 ${PCL_COMMON_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES})

#add_executable(StereoCalibration calibrate.cpp ${CLASSES})
#target_compile_features(StereoCalibration PRIVATE cxx_range_for)
#target_link_libraries(StereoCalibration ${OPENGL_LIBRARIES} ${OpenCV_LIBS} ${GLUT_LIBRARIES} v4l2)

add_executable(VideoMaker video.cpp ${CLASSES})
target_compile_features(VideoMaker PRIVATE cxx_range_for)
target_link_libraries(VideoMaker ${OpenCV_LIBS} v4l2)

add_executable(SceneRenderer SceneRenderer.cpp ${CLASSES})
target_compile_features(SceneRenderer PRIVATE cxx_range_for)
target_link_libraries(SceneRenderer ${OpenCV_LIBS} v4l2)

#add_executable(TriTest __unused/TriTest.cpp ${CLASSES} __unused/VirtualCamera.cpp __unused/FakeCamera.cpp)
#target_compile_features(TriTest PRIVATE cxx_range_for)
#target_link_libraries(TriTest ${OpenCV_LIBS} v4l2 ${PCL_LIBRARIES})

add_executable(OmpTest omp_test.cpp)

