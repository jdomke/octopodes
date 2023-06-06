#include "../../include/envs/MountainCar.hpp"
#include <iostream>

using namespace octorl;

MountainCar::MountainCar() {
    //action_space = DiscreteActionSpace(3);
    
    std::default_random_engine gen;
    std::uniform_real_distribution<float> distribution(-0.6,-0.4);

    position = distribution(gen);

    velocity = 0;
    steps = 0;
}

void MountainCar::transition(int action) {
    velocity = velocity + (action - 1)*FORCE - std::cos(3*position)*GRAVITY;
    velocity = std::min((float) std::max((float)-0.07,velocity),(float) 0.07);

    position = position + velocity;
    position = std::min((float) std::max((float) -1.2, position), (float) 0.6);
    if(position == -1.2)
        velocity = 0;
}

StepReturn MountainCar::step(int action) {
    StepReturn obs;
    obs.reward = -1;
    transition(action);
    if(steps + 1 == 200)
        obs.done = true;
    else if(position >= 0.5)
        obs.reward = 0;
    
    steps++;

    return obs;
}

float MountainCar::getPosition() {
    return position;
}