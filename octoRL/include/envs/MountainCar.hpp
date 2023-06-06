#ifndef MOUNTAIN_CAR_H
#define MOUNTAIN_CAR_H

#include "EnvironmentsBase.hpp"
#include <bits/stdc++.h>
#include <cmath>
#include <algorithm>      
namespace octorl {

    class MountainCar : public EnvironmentsBase {
    private:
        DiscreteActionSpace action_space;
        const float FORCE = 0.001;
        const float GRAVITY = 0.0025;
        float position;
        float velocity;
        int steps;
        void transition(int action);
    public:
        MountainCar();
        //~MountainCar();

        StepReturn step(int action);
        float getPosition();

    };
}

#endif