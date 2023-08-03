#include "../../include/envs/CNNTest.hpp"
#include <iostream>
#include <time.h>

octorl::CNNTest::CNNTest() {
    reset();
}

octorl::CNNTest::CNNTest(int s) {
    seed = s;
    
    reset();

}

octorl::StepReturn octorl::CNNTest::step(int action) {
    steps++;
    StepReturn obs;
    obs.reward = -1;

    if(steps > 100) 
        obs.done = true;
    obs.observation = torch::rand({1,3,32,32});
    return obs;
}


octorl::StepReturn octorl::CNNTest::reset() {
    steps = 0;
    StepReturn obs;
    obs.reward = -1;

    if(steps > 100) 
        obs.done = true;
    obs.observation = torch::rand({1,3,32,32});
    return obs;
}

int octorl::CNNTest::getActionSize() {
    return action_space_size;
}

int octorl::CNNTest::getObservationSize() {
    return observation_space_size;
}

int octorl::CNNTest::currentStep() {
    return steps;
}

torch::Tensor octorl::CNNTest::shapeObservation(torch::Tensor buffer){
    return buffer.reshape({1,3,32,32}); // may need to change
}

torch::Tensor octorl::CNNTest::obsBuffer(int b){
    return torch::zeros({b,3,32,32});
}

torch::Tensor octorl::CNNTest::getState() {
    return torch::zeros({1,3,32,32});
}