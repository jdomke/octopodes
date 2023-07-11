#ifndef POLICY_BASE_H
#define POLICY_BASE_H

#include <torch/torch.h>

namespace octorl {
    

    class PolicyBase : public torch::nn::Module {
        public:
            virtual torch::Tensor forward(torch::Tensor x) = 0;
           // virtual PolicyBase& operator= (const PolicyBase& p) = 0;
            virtual void serialize(float *buffer) = 0;
            virtual void loadFromSerial(float *buffer) = 0;
            virtual void applyGradient(float *buffer, int num_steps = 1)= 0;
            virtual int getElementCount() = 0;
        private:
            int num_elem_param; 
    };

}
#endif