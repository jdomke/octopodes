#include "../include/Spaces.hpp"

using namespace octorl;

DiscreteActionSpace::DiscreteActionSpace(int act_size) {
    actions = new int(act_size);
}

DiscreteActionSpace::~DiscreteActionSpace() {
    delete actions;
}

