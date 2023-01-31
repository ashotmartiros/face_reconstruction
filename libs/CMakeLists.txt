cmake_minimum_required(VERSION 3.15)

set(LIB_SOURCES  Eigen.h                         Renderer.cpp
                 BFM.h                           DenseOptimizer.cpp
                 LandmarkDetection.h
                 Renderer.h
                 DenseOptimizer.h
)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)

set(GLFW_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glfw/include CACHE PATH "Path to glfw include")
set(GLFW_LIBRARY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glfw/lib-x86_64/libglfw.3.dylib CACHE PATH "Path to glfw include")

set(GLEW_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glew/include CACHE PATH "Path to glew include")
set(GLEW_LIBRARY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glew/lib/libGLEW.a CACHE PATH "Path to glew libraries")

set(FreeImage_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/FreeImage CACHE PATH "Path to FreeImage header file")
set(FreeImage_LIBRARY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/FreeImage/libfreeimage.a CACHE PATH "Path to FreeImage .lib/.dll folder")

add_library(FRLibrary ${LIB_SOURCES})
target_include_directories (FRLibrary PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} 
                                             ${FreeImage_INCLUDE_DIR} 
                                             ${GLEW_INCLUDE_DIRS} 
                                             ${GLFW_INCLUDE_DIRS} 
                                             ${OpenCV_INCLUDE_DIRS}
                                             ${dlib_INCLUDE_DIR}
                                              )
target_link_libraries(FRLibrary Eigen3::Eigen ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OpenCV_LIBS} dlib::dlib) 
