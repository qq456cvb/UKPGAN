cmake_minimum_required(VERSION 2.8.12)
project(shot)
set (CMAKE_CXX_STANDARD 11)

find_package( PythonInterp 3.6 REQUIRED )
find_package( PythonLibs 3.6 REQUIRED )
find_package(pybind11 REQUIRED)
find_package( PCL 1.8 REQUIRED )

include_directories( ${PCL_INCLUDE_DIRS} )
# link_directories( ${PCL_LIBRARY_DIRS} )
add_definitions(${PCL_DEFINITIONS})

pybind11_add_module(shot shot.cpp)
target_link_libraries(shot PUBLIC ${PCL_LIBRARIES})