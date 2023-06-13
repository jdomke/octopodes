#include "../include/Mlp.hpp"
#include "Mlp.hpp"
// for now only using relu

octorl::Mlp::Mlp(int input, int output, int hidden_layers, int *nodes) {
    
    std::string label = "fc";
    input_size = input;
    output_size = output;
    total_layers = hidden_layers + 1;   

    if(hidden_layers > 0) {
        int i = 0;
        layers.push_back(register_module("input",torch::nn::Linear(input,nodes[i++])));
        for(i; i < hidden_layers;i++) 
            layers.push_back(register_module(label + (char)(48 + i),torch::nn::Linear(nodes[i-1],nodes[i])));
        layers.push_back(register_module("output",torch::nn::Linear(nodes[i-1], output)));
    }
    else    
        layers.push_back(register_module(label,torch::nn::Linear(input,output)));
    
}


torch::Tensor octorl::Mlp::forward(torch::Tensor x)
{

    int i = 0;
    //x = x.reshape({x.size(-1),x.size(0)});
    for(i; i < total_layers-1; i++) {
        x = torch::relu(layers[i]->forward(x));
    }
    x = layers.back()->forward(x);
    return x;
}

octorl::Mlp& octorl::Mlp::operator= (const octorl::Mlp& mlp){
    input_size = mlp.input_size;
    output_size = mlp.output_size;
    total_layers = mlp.total_layers;

    int i = 0;
    for(auto l : mlp.layers){
        //std::cout<<l[0]<<std::endl;
        layers.push_back(register_module("layer" + (char)(48 + i++) ,l));
    }
    return *this;
}


// https://discuss.pytorch.org/t/how-to-copy-network-parameters-in-libtorch-c-api/32221
void octorl::loadstatedict(torch::nn::Module& model, torch::nn::Module& target_model) {
torch::autograd::GradMode::set_enabled(false);  // make parameters copying possible
auto new_params = target_model.named_parameters(); // implement this
auto params = model.named_parameters(true /*recurse*/);
auto buffers = model.named_buffers(true /*recurse*/);
for (auto& val : new_params) {
    auto name = val.key();
    
    auto* t = params.find(name);
    if (t != nullptr) {
        t->copy_(val.value());
    } else {
        t = buffers.find(name);
        if (t != nullptr) {
            t->copy_(val.value());
            }
        }
    }
    torch::autograd::GradMode::set_enabled(true);
}