#ifndef POLICY_H
#define POLICY_H

#include <torch/torch.h>
#include <vector>
#include <string>
#include <utility>
#include <memory>
#include <time.h>
//#include "PolicyBase.hpp"

// used to easily generate test networks, it is better to just define your own class

namespace octorl{

    enum layer_type {
        linear,
        conv2d,
        max_pool_2d,
        flatten
    };

    enum activation_type {
        relu,
        softmax,
        none,
        sigmoid,
        tanh

    };


    // only supporting symmetrical kernal, padding, dilation
    struct LayerInfo {
        layer_type type;
        activation_type activation;
        std::string label;
        int vect_position;
        int input, output, kernel_size, stride, padding, dilation;
        LayerInfo(layer_type t, activation_type a, std::string l, int i, int out, 
            int ks = 1, int s = 1, int p = 0, int d = 1);
    };

    class Policy : public torch::nn::Module  {
        
        public:
            Policy(std::vector<LayerInfo> l_i,int s = (int)time(NULL));
            Policy() {};
            torch::Tensor forward(torch::Tensor x);
            Policy& operator= (const Policy& p);
            void serialize(float *buffer);
            void loadFromSerial(float *buffer);
            void applyGradient(float *buffer);
            int getElementCount();
            
            std::vector<torch::nn::Linear> linear_layers;
            std::vector<torch::nn::Conv2d> conv2d_layers;
            std::vector<torch::nn::MaxPool2d> pool2d_layers;
            std::vector<torch::nn::Flatten> flatten_layers;
        private:
            //std::vector<std::shared_ptr<torch::nn::AnyModule>> layers;
            //std::vector<std::shared_ptr<torch::nn::ModuleHolder>> layers;
            //torch::nn::ModuleList layers;
            std::vector<LayerInfo> layer_info;
            torch::Tensor activation(torch::Tensor x, activation_type act);
            void addLayer(LayerInfo &l);
            int num_elem_param = 0; 
            int seed;
            // need a vector for each layer type

    };
    
    //void loadstatedict(torch::nn::Module& model, torch::nn::Module& target_model);
} 

#endif