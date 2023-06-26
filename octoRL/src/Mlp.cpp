#include "../include/Mlp.hpp"
#include "Mlp.hpp"
// for now only using relu

octorl::Mlp::Mlp(int input, int output, int hidden_layers, int *nodes) {
    
    std::string label = "fc";
    input_size = input;
    output_size = output;
    total_layers = hidden_layers + 1;   
    num_elem_param = 0;

    if(hidden_layers > 0) {
        int i = 0;
        layers.push_back(register_module("input",torch::nn::Linear(input,nodes[i])));
        num_elem_param += input*nodes[i] + nodes[i];
        layer_shapes.push_back(std::make_pair(input_size, nodes[i++]));
        for(i; i < hidden_layers;i++)  {
            layers.push_back(register_module(label + (char)(48 + i),torch::nn::Linear(nodes[i-1],nodes[i])));
            layer_shapes.push_back(std::make_pair(nodes[i-1], nodes[i]));
            num_elem_param += nodes[i-1]*nodes[i] + nodes[i];
        }
        layers.push_back(register_module("output",torch::nn::Linear(nodes[i-1], output)));
        layer_shapes.push_back(std::make_pair(nodes[i-1], output));
        //layer_shapes.push_back(std::make_pair(1, output));
        num_elem_param += output*nodes[i-1] + output;
    }
    else {
        layers.push_back(register_module(label,torch::nn::Linear(input,output)));
        layer_shapes.push_back(std::make_pair(input_size, output)); 
        num_elem_param += input_size * output + output;
    }
    

   
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
    num_elem_param = mlp.num_elem_param;


    int i = 0;
    for(auto l : mlp.layer_shapes) 
        layers.push_back(register_module("layer" + (char)(48 + i++),torch::nn::Linear(l.first, l.second)));

    layer_shapes = mlp.layer_shapes;
    return *this;
    // for(auto l : mlp.layers){
    //     //std::cout<<l[0]<<std::endl;
    //     layers.push_back(register_module("layer" + (char)(48 + i++) ,l));
    // }
    // layer_shapes = mlp.layer_shapes;
    // return *this;
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



// can make this better
//https://github.com/soumyadipghosh/eventgrad/blob/master/dmnist/decent/decent.cpp#L96
void octorl::Mlp::serialize(float *buffer) {
    auto sz = named_parameters().size();
    auto params = named_parameters();

    auto param_elem_size = params[0].value().element_size();
    //std::cout<<"model  Count "<<num_elem_param<<std::endl;
    
   // std::shared_ptr<float> serial_params(new float[num_elem_param]);

    for (int i = 0; i < num_elem_param; i++)
        buffer[i] = 0.0;


    int sp = 0;
    //std::cout<<sz<<std::endl;
    for(auto i = 0; i < sz; i++) {
        auto flat = torch::flatten(params[i].value());

        for(int j = 0; j < flat.numel(); j++)
            buffer[sp++] = flat[j].item<float>();
    }

    //return serial_params;
}

void octorl::Mlp::loadFromSerial(float *buffer) {
    torch::autograd::GradMode::set_enabled(false);
    auto p = parameters();
    int b =0,pC =0;
    for(int i =0; i < layer_shapes.size(); i++) {
        
        for (int j = 0; j < layer_shapes[i].second; j++)
        {
            if(layer_shapes[i].first == 1) {
                p[pC][j] = buffer[b++];
            }
            else {
                for (int k = 0; k < layer_shapes[i].first; k++)
                {
                    p[pC][j][k] = buffer[b++];
                }
            }
        } 
        //biases
        pC += 1;
        for (int j = 0; j < layer_shapes[i].second; j++) {
            p[pC][j] = buffer[b++];

        }
        pC += 1;
    }
    torch::autograd::GradMode::set_enabled(true);
}


int octorl::Mlp::getElementCount() {
    return num_elem_param;
}