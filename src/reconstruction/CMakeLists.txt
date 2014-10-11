cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(Reconstruction)

find_package(OpenCV 3.0.0 REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(save_images save_images.cpp Canvas.cpp Magic.cpp)
target_link_libraries(save_images ${OpenCV_LIBS})

configure_file(../stereo_calib.xml stereo_calib.xml COPYONLY)
add_executable(stereo_calib stereo_calib.cpp)
target_link_libraries(stereo_calib ${OpenCV_LIBS})

add_executable(merge_clouds merge_clouds.cpp Magic.cpp PCDReader.cpp PCDWriter.cpp)
target_link_libraries(merge_clouds ${OpenCV_LIBS})

configure_file(../intrinsics.yml intrinsics.yml COPYONLY)
configure_file(../extrinsics.yml extrinsics.yml COPYONLY)
configure_file(../left01.jpg left01.jpg COPYONLY)
configure_file(../right01.jpg right01.jpg COPYONLY)
add_executable(reconstruct reconstruct.cpp Camera.cpp Canvas.cpp Magic.cpp OpenGLRenderer.cpp PCDWriter.cpp PointCloudSegmenter.cpp)
target_link_libraries(reconstruct ${OPENGL_LIBRARIES} ${OpenCV_LIBS})

configure_file(../left01_bal.jpg left01_bal.jpg COPYONLY)
configure_file(../right01_bal.jpg right01_bal.jpg COPYONLY)
add_executable(triangulate triangulate.cpp Camera.cpp Canvas.cpp Magic.cpp OpenGLRenderer.cpp PCDWriter.cpp PointCloudSegmenter.cpp)
target_link_libraries(triangulate ${OPENGL_LIBRARIES} ${OpenCV_LIBS})
