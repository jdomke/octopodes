#include "../../include/envs/Cartpole.hpp"
#include <iostream>
octorl::Cartpole::Cartpole() {
    reset();    
}
octorl::Cartpole::Cartpole(int seed) {
    gen.seed(seed);
    reset();    
}

octorl::StepReturn octorl::Cartpole::step(int action) {
    //std::cout<<"in step\n";
    //std::cout<<"x: "<<x<<", theta: "<<theta<<std::endl; 
    octorl::StepReturn obs;
    float force;
    if(action == 1)
        force = FORCE_MSG;
    else
        force = -FORCE_MSG;

    float cos_theta = std::cos(theta);
    float sin_theta = std::sin(theta);

    float temp = (force + POLEMASS_LENGTH * (theta_dot*theta_dot) * sin_theta) / TOTAL_MASS;
    float theta_acc = (GRAVITY * sin_theta - cos_theta * temp) / (LENGTH * (4.0/3.0 - MASSPOLE * (cos_theta*cos_theta)/TOTAL_MASS));
    float x_acc = temp - POLEMASS_LENGTH * theta_acc * cos_theta / TOTAL_MASS;

    x = x + TAU * x_dot;
    x_dot = x_dot + TAU * x_acc;
    theta = theta + TAU * theta_dot;
    theta_dot = theta_dot + TAU * theta_acc;
    //std::cout<<"post\n";
    //std::cout<<"x: "<<x<<", theta: "<<theta<<std::endl; 
    obs.terminated = (x < -X_THRESHOLD) || (x > X_THRESHOLD) || (theta < -THETA_THRESHOLD_RADIANS) || (theta > THETA_THRESHOLD_RADIANS);

    if(!obs.terminated)
        obs.reward = 1;
    else {
        obs.done = true;
        obs.reward = 1;
    }
    if(steps >= 500) {
        obs.done = true;
        obs.goal = true;
        obs.reward = 1;
    }

    obs.observation = torch::tensor({{x, x_dot, theta, theta_dot}});
    steps++;
    return obs;
}

octorl::StepReturn octorl::Cartpole::reset() {
    octorl::StepReturn obs;
    steps = 0;
    x = starting_gen(gen);//position_gen(gen);
    x_dot = starting_gen(gen);//velocity_gen(gen);
    theta = starting_gen(gen);//theta_gen(gen);
    theta_dot = starting_gen(gen);//velocity_gen(gen);

    obs.observation = torch::tensor({{x, x_dot, theta, theta_dot}});

    return obs;
}

float octorl::Cartpole::getPosition() {
    return x;
}

int octorl::Cartpole::getActionSize() {
    return action_space_size;
}

int octorl::Cartpole::getObservationSize() {
    return observation_space_size;
}

int octorl::Cartpole::currentStep() {
    return steps;
}

void octorl::Cartpole::setState(float xset, float x_dotset, float thetaset, float theta_dotset) {
    x = xset;
    x_dot = x_dotset;
    theta = thetaset;
    theta_dot = theta_dotset;
}


int octorl::Cartpole::memorySize() {

    return 2*observation_space_size + 3;
}

torch::Tensor octorl::Cartpole::getState(){
    return torch::tensor({{x, x_dot, theta, theta_dot}});
}