#include <iostream>
#include "include/envs/MountainCar.hpp"


using namespace std;

int main() {

    octorl::MountainCar env;
    octorl::StepReturn obs;
    int action;

    while(1) {
        cin >> action;

        obs = env.step(action);

        if(obs.done)
            return 0;

        cout<<env.getPosition()<<endl;
    }
    return 0;
}