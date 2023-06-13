#ifndef AGENT_BASE_H
#define AGENT_BASE_H

#include <torch/torch.h>
#include "Spaces.hpp"

namespace octorl {

    class AgentBase {
        public: 
            virtual int action(torch::Tensor obs) = 0;

        private:
           


        protected: 
            AgentBase();
    };
}

#endif