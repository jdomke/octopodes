#ifndef MLP_H
#define MLP_H

#include <torch/torch.h>
#include <vector>
#include <string>
namespace octorl {
    
    struct Mlp : torch::nn::Module {
        public:
            Mlp(int input, int output, int hidden_layers, int *nodes); 
            Mlp(){};
            torch::Tensor forward(torch::Tensor x);
            Mlp& operator= (const Mlp& mlp);
        private:
            int input_size, output_size, total_layers;
            std::vector<torch::nn::Linear> layers;
    };

    void loadstatedict(torch::nn::Module& model, torch::nn::Module& target_model);
}

#endif