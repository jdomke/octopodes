cmake_minimum_required(VERSION 3.16)
SET(CMAKE_CUDA_STANDARD 11)
project(OctoRL
    VERSION 0.01
    DESCRIPTION "RL OCTOPODE"
    LANGUAGES CXX)
find_package(Torch REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)


find_package(libconfig REQUIRED)

find_library(libconfig config)
find_library(libconfig libconfig)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(PAPI REQUIRED)
include_directories(${PAPI_INCLUDE_DIRS})


add_executable(main main.cpp
    #src/driver.cpp
    src/Spaces.cpp
    src/Mlp.cpp
    src/envs/MountainCar.cpp
    src/ExperienceReplay.cpp
    #src/agents/Dqn.cpp
    src/envs/Blackjack.cpp
    src/envs/Cartpole.cpp
    src/envs/CNNTest.cpp
    #src/quickTest.cpp
    src/agents/DqnAsync.cpp
    src/agents/A3C.cpp
    src/agents/A2C.cpp
    #src/Cnn.cpp
    src/Policy.cpp
    src/helper.cpp
    src/configure.cpp)  

# if(OpenMP_CXX_FOUND)
#     target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)
# endif() 
#add_compile_options(-lconfig++)
#add_compile_options(-lconfig) config++ config 
target_link_libraries(main "${TORCH_LIBRARIES}" OpenMP::OpenMP_CXX MPI::MPI_CXX config config++ ${PAPI_LIBRARIES}) 
#add_compile_options(-openmp -lpthread)
#target_link_options(main PUBLIC ${OpenMP_CXX_FLAGS})


set_property(TARGET main PROPERTY CXX_STANDARD 17)
target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET main
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:main>)
endif (MSVC)
