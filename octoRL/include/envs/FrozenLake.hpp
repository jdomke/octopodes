#ifndef FROZEN_LAKE_H
#define FROZEN_LAKE_H

#include "EnvironmentsBase.hpp"
#include <random>
#include <cmath>
#include <string>
#include <algorithm>      


namespace octorl {

    class FrozenLake : virtual public EnvironmentsBase
    private:
        int player_x;
        int player_y;
        std::string map1[4]{"SFFF", "FHFH", "FFFH", "HFFG"};
    public:
        FrozenLake();
        StepReturn step(int action);
        StepReturn reset();
        float getPosition();
        int getActionSize();
        int getObservationSize();
        int currentStep();
}


#endif