#include "../include/Cnn.hpp"




octorl::Cnn2d::Cnn2d() {

    fc1 = register_module("input", torch::nn::Conv2d(4,3,2));
    fc2 = register_module("layer1", torch::nn::Conv2d(3,3,2));
}