#ifndef CNN_H
#define CNN_H

#include <torch/torch.h>
#include <vector>
#include <string>
#include <utility>
#include <memory>

namespace octorl {

    struct Cnn2d : torch::nn::Module {
        public:

            Cnn2d();
            torch::nn::Conv2d fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
            torch::nn::Linear out{nullptr};
        private:
            torch::Tensor forward(torch::Tensor x){return x;};
            std::vector<std::shared_ptr<torch::nn::Module>> layers;

    };
}
#endif  