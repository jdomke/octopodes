#include "../include/Spaces.hpp"

octorl::DiscreteActionSpace::DiscreteActionSpace(int act_size) {
    actions = new int(act_size);
}

octorl::DiscreteActionSpace::~DiscreteActionSpace() {
    delete actions;
}

octorl::ObservationSpace::ObservationSpace(int size, int dim) {
    
}