cmake_minimum_required(VERSION 3.5)
project(plane_projection)

# 不需要 find_package(OpenCV) 了

file(GLOB_RECURSE EXAMPLE_SRC "plane_projection.cpp")

add_executable(plane_projection ${EXAMPLE_SRC})

# 只要連結 ezsift 即可 (若有用到數學庫可能需要 -lm，但在 CMake 通常自動處理)
target_link_libraries(plane_projection
    ezsift
)