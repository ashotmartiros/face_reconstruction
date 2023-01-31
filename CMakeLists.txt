cmake_minimum_required(VERSION 3.15)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)

project(face_reconstruction)

# Set C++ flags
set(CMAKE_CXX_STANDARD 14)

# third party dependencies
find_package(eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(glog REQUIRED ${glog_DIR_HINT})
find_package(dlib REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)

# OpenGL
set(GLFW_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glfw/include CACHE PATH "Path to glfw include")
set(GLFW_LIBRARY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glfw/lib-x86_64/libglfw.3.dylib CACHE PATH "Path to glfw include")

set(GLEW_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glew/include CACHE PATH "Path to glew include")
set(GLEW_LIBRARY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glew/lib/libGLEW.a CACHE PATH "Path to glew libraries")

set(FreeImage_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/FreeImage CACHE PATH "Path to FreeImage header file")
set(FreeImage_LIBRARY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/FreeImage CACHE PATH "Path to FreeImage .lib/.dll folder")

#set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${Ceres_DIR})

add_definitions("-D_DISABLE_EXTENDED_ALIGNED_STORAGE")

#include(files.cmake)
add_subdirectory(libs)
include_directories(libs)

link_directories(${FreeImage_LIBRARY})

add_executable(face_reconstruction main.cpp)
target_link_libraries(face_reconstruction FRLibrary 
                                          Eigen3::Eigen
                                          ceres 
                                          freeimage 
                                          dlib::dlib 
                                          ${OpenCV_LIBS} 
                                          ${GLEW_LIBRARY} 
                                          ${GLFW_LIBRARY} 
                                          ${OPENGL_LIBRARIES})

target_include_directories(face_reconstruction PUBLIC 
                                               FRLibrary
                                               ${CMAKE_CURRENT_SOURCE_DIR}  
                                               ${EIGEN_INCLUDE_DIRS}
                                               ${Ceres_INCLUDE_DIR} 
                                               ${glog_INCLUDE_DIR} 
                                               ${FreeImage_INCLUDE_DIR} 
                                               ${OpenCV_INCLUDE_DIRS} 
                                               ${dlib_INCLUDE_DIR} 
                                               ${GLEW_INCLUDE_DIRS} 
                                               ${GLFW_INCLUDE_DIRS})

