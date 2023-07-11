#ifndef HELPER_H
#define HELPER_H

#include <torch/torch.h>
#include "Policy.hpp"

namespace octorl {
    void loadstatedict(std::shared_ptr<Policy>  model, std::shared_ptr<Policy> target_model);
    void loadstatedict2(Policy&  model, Policy& target_model);
}

#endif