#include "../include/Policy.hpp"


octorl::LayerInfo::LayerInfo(octorl::layer_type t, octorl::activation_type a, std::string l, int i, int out, 
            int ks, int p, int d) {
    type = t;
    activation = a;
    label = l;
    input = i;
    output = out;
    kernel_size = ks;
    padding = p;
    dilation = d;
}

octorl::Policy::Policy(std::vector<octorl::LayerInfo> l_i) {
    layer_info = l_i;
    for(auto &l : layer_info) {
        addLayer(l);
    }
}

torch::Tensor octorl::Policy::forward(torch::Tensor x) {

    int i = 0;

    for(i; i < layer_info.size(); i++) {
        //x = activation(layers[i]->as<torch::nn::Linear>()->forward(x),layer_info[i].activation);
       // x = activation(layers[i]->forward(x),layer_info[i].activation);
        switch (layer_info[i].type) {
            case linear:
                x = activation(linear_layers[layer_info[i].vect_position]->forward(x),layer_info[i].activation);
               // std::cout<<layer_info[i].label<<" "<<x<<std::endl;
                break;
            case conv2d:
                x = activation(conv2d_layers[layer_info[i].vect_position]->forward(x),layer_info[i].activation);
             //   std::cout<<layer_info[i].label<<" "<<x<<std::endl;
                break;
            case max_pool_2d:
                x = activation(pool2d_layers[layer_info[i].vect_position]->forward(x),layer_info[i].activation);
                // std::cout<<layer_info[i].label<<" "<<x<<std::endl;
                break;
            case flatten:
                x = activation(flatten_layers[layer_info[i].vect_position]->forward(x),layer_info[i].activation);
                //x = x.reshape({x.size(-1), x.size(0)});
                //std::cout<<layer_info[i].label<<" "<<x<<std::endl;
                break;
            default:
                break;
        }
    }
    return x;
}

torch::Tensor octorl::Policy::activation(torch::Tensor x, octorl::activation_type act) {
    switch (act) {
        case relu:
            return torch::relu(x).requires_grad_(true);
            break;
        case softmax:
            return torch::softmax(x, 1).requires_grad_(true);
            break;
        default:
            return x;
            break;
    }
}

void octorl::Policy::addLayer(octorl::LayerInfo& l) {
    switch (l.type) {
        case linear:
            linear_layers.push_back(register_module(l.label,torch::nn::Linear(l.input, l.output)));
            num_elem_param += l.input * l.output + l.output;
            l.vect_position = linear_layers.size() - 1;
            break;
        case conv2d:
            conv2d_layers.push_back(register_module(l.label,torch::nn::Conv2d(l.input, l.output, l.kernel_size)));
            num_elem_param += l.input * l.output * l.kernel_size * l.kernel_size + l.output;
            l.vect_position = conv2d_layers.size() - 1;
            break;
        case flatten:
            flatten_layers.push_back(register_module(l.label,
                torch::nn::Flatten()));
            l.vect_position = flatten_layers.size() - 1;
            break;
        case max_pool_2d: 
            pool2d_layers.push_back(register_module(l.label, torch::nn::MaxPool2d(l.kernel_size)));
            l.vect_position = pool2d_layers.size() - 1;
            break;
        default:
            break;
    }
}

octorl::Policy& octorl::Policy::operator= (const octorl::Policy& p) {
    layer_info = p.layer_info;
    for(auto &l : layer_info) {
        addLayer(l);
    }
    return *this;
}

void octorl::Policy::serialize(float *buffer) {
    auto sz = named_parameters().size();
    auto params = named_parameters();

    for (int i = 0; i < num_elem_param; i++)
        buffer[i] = 0.0;

    int sp = 0;
    for(auto i = 0; i < sz; i++) {
        auto flat = torch::flatten(params[i].value());

        for(int j = 0; j < flat.numel(); j++)
            buffer[sp++] = flat[j].item<float>();
    }
}


// rework

void octorl::Policy::loadFromSerial(float *buffer){
    torch::autograd::GradMode::set_enabled(false);
    auto p = parameters();
    int b =0,pC =0;
    for(int l = 0; l < layer_info.size(); l++) {
        auto sizes = p[pC].sizes();

        switch(layer_info[l].type) {
            case octorl::linear:
                
                for(int i = 0; i < sizes[0]; i++) {
                    for(int j = 0; j < sizes[1]; j++) {
                        p[pC][i][j] = buffer[b++];
                    }
                }
        
                sizes = p[++pC].sizes();

                for(int i = 0; i < sizes[0]; i++)
                    p[pC][i] = buffer[b++];
                pC++;
                break;
            case octorl::conv2d:
                for(int i = 0; i < sizes[0]; i++) {
                    for(int j = 0; j < sizes[1]; j++) {
                        for(int k = 0; k < sizes[2]; k++) {
                            for(int m = 0; m < sizes[3]; m++) {
                                p[pC][i][j][k][m] = buffer[b++];
                            }
                        }
                    }
                }
            
                sizes = p[++pC].sizes();

                for(int i = 0; i < sizes[0]; i++)
                    p[pC][i] = buffer[b++];
                pC++;
                break;
            default:
                break;

        }
    }
    torch::autograd::GradMode::set_enabled(true);

}

void octorl::Policy::applyGradient(float *buffer, int num_steps)
{
    int b = 0, pc = 0;
    auto p = parameters();
    for(int l = 0; l < layer_info.size(); l++) {
        auto sizes = p[pc].sizes();

        auto g = torch::zeros(sizes);

        switch(layer_info[l].type) {
            case octorl::linear:
                for(int i = 0; i < sizes[0]; i++) {
                    for(int j = 0; j < sizes[1]; j++) {
                        g[i][j] = buffer[b++];
                    }
                }
                parameters()[pc].mutable_grad() = g;
                sizes = p[++pc].sizes();
                g = torch::zeros(sizes);
                for(int i = 0; i < sizes[0]; i++)
                    g[i] = buffer[b++];
                parameters()[pc].mutable_grad() = g;
                pc++;
                break;
            case octorl::conv2d:
                for(int i = 0; i < sizes[0]; i++) {
                    for(int j = 0; j < sizes[1]; j++) {
                        for(int k = 0; k < sizes[2]; k++) {
                            for(int m = 0; m < sizes[3]; m++) {
                                g[i][j][k][m] = buffer[b++];
                            }
                        }
                    }
                }
            
                parameters()[pc].mutable_grad() = g;
                sizes = p[++pc].sizes();
                g = torch::zeros(sizes);

                for(int i = 0; i < sizes[0]; i++)
                    g[i] = buffer[b++];
                parameters()[pc].mutable_grad() = g;
                pc++;
                break;
            default:
                break;
        }
    }

}


int octorl::Policy::getElementCount(){ return num_elem_param;}
