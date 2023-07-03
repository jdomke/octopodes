#include "../../include/envs/MountainCar.hpp"
#include <iostream>
#include <time.h>

octorl::MountainCar::MountainCar() {
    reset();
}
octorl::MountainCar::MountainCar(int seed) {
    //action_space = DiscreteActionSpace(3);
    gen.seed(seed);
    reset();
}

void octorl::MountainCar::transition(int action) {
  //  std::cout<<"pre transition: "<<position<<" "<<velocity<<std::endl;
    velocity = velocity + (action - 1)*FORCE - std::cos(3*position)*-GRAVITY;
    velocity = std::min((float) std::max((float)-0.07,velocity),(float) 0.07);
  
    position = position + velocity;
    //std::cout<<"post transition: "<<position<<" "<<velocity<<std::endl;
    position = std::min((float) std::max((float) -1.2, position), (float) 0.6);
    if(position < -1.2){ 
        velocity = 0;
        position = -1.2;
    }
    if(position > 0.6){ 
        velocity = 0;
        position = 0.6;
    }
    //std::cout<<"post rounding transition: "<<position<<" "<<velocity<<std::endl;
}

octorl::StepReturn octorl::MountainCar::step(int action) {
    StepReturn obs;
    obs.reward = -1;
    ///std::cout<<action<<std::endl;
    transition(action);
    //double obs_vals[] = {position,velocity};
    if(steps + 1 >= 200)
        obs.done = true;
    if(position >= 0.5) {
        obs.reward = 10;
        obs.goal = true;
        obs.done = true;
    }
    else if(position > -0.6)
        obs.reward = std::pow(1+position,2);
       
    
    obs.observation = torch::tensor({{position,velocity}});
    //std::cout<<steps<<" ";
    steps++;

    return obs;
}

float octorl::MountainCar::getPosition() {
    return position;
}

octorl::StepReturn octorl::MountainCar::reset() {
    
    StepReturn obs;

    position = distribution(gen);
    velocity = 0;
    steps = 0;
    obs.observation = torch::tensor({{position,velocity}});
    return obs;
}

int octorl::MountainCar::getActionSize() {
    return action_space_size;
}

int octorl::MountainCar::getObservationSize() {
    return observation_space_size;
}

int octorl::MountainCar::currentStep() {
    return steps;
}

void octorl::MountainCar::setState(float p, float v) {
    position = p;
    velocity = v;
}

int octorl::MountainCar::memorySize() {

    return 2*observation_space_size + 3;
}


torch::Tensor octorl::MountainCar::getState(){
    return torch::tensor({{position,velocity}});
}