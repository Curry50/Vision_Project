cmake_minimum_required(VERSION 3.22)
project(ImgTest)

set(CMAKE_CXX_STANDARD 14)
aux_source_directory(src MY_LIBRARY_DIRS)
add_library(MyLibrary SHARED ${MY_LIBRARY_DIRS} )
target_include_directories(MyLibrary PUBLIC include)

add_executable(ImgTest src/main.cpp )
find_package(OpenCV REQUIRED)
target_link_libraries(ImgTest MyLibrary)
target_link_libraries(MyLibrary ${OpenCV_LIBS})
target_link_libraries(ImgTest ${OpenCV_LIBS})