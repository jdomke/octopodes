
#include "../include/helper.hpp"


void octorl::loadstatedict(std::shared_ptr<octorl::Policy>  model, std::shared_ptr<octorl::Policy> target_model) {
    torch::autograd::GradMode::set_enabled(false);  // make parameters copying possible
    auto new_params = target_model->named_parameters(); // implement this
    auto params = model->named_parameters(true /*recurse*/);
    auto buffers = model->named_buffers(true /*recurse*/);
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

void octorl::loadstatedict2(octorl::Policy& model, octorl::Policy& target_model) {
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