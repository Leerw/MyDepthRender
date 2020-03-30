cmake_minimum_required(VERSION 2.8)

project(myrender)

find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(glm REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)

include_directories("${PROJECT_SOURCE_DIR}/include/GLFW")
include_directories(${GLUT_INCLUDE_DIR})
include_directories(${OpenGL_INCLUDE_DIR})
include_directories(${GLEW_INCLUDE_DIR})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRECTORIES})
include_directories(${GLM_INCLUDE_DIRS})

link_directories(${GLM_LIBRARY_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

add_executable(render main.cpp)
target_link_libraries(render ${OpenGL_LIBRARIES} ${GLUT_LIBRARY} ${GLEW_LIBRARY} ${GLM_LIBRARIES} ${OpenCV_LIBS} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES})