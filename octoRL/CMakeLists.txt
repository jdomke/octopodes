cmake_minimum_required(VERSION 3.16)


project(OctoRL
    VERSION 0.01
    DESCRIPTION "RL OCTOPODE"
    LANGUAGES CXX)

add_executable(main main.cpp
    src/Spaces.cpp
    src/envs/MountainCar.cpp)   
    #include_directories(${PROJECT_SOURCE_DIR})
    
    #add_subdirectory(src)
target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)