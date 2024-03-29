cmake_minimum_required(VERSION 3.1)
project(kinematics)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

### libIGL options: choose between header only and compiled static library
option(LIBIGL_USE_STATIC_LIBRARY     "Use libigl as static library" OFF)
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)

find_package(LIBIGL REQUIRED QUIET)

# Add your project files
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include/")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/json/")
if(USE_SOLUTION)
  file(GLOB SRCFILES solution/*.cpp)
else()
  file(GLOB SRCFILES src/*.cpp)
endif()

add_executable(${PROJECT_NAME} "main.cpp" ${SRCFILES})
add_executable(random_test random_test.cpp ${SRCFILES})
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/shared/json)
target_link_libraries(${PROJECT_NAME} igl::core igl::opengl igl::opengl_glfw )
target_include_directories(random_test SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/shared/json)
target_link_libraries(random_test igl::core igl::opengl igl::opengl_glfw )